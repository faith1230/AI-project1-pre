import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import torch

from configs.base_config import BaseConfig
from src.dqn_agent import DQNAgent
from src.environment import describe_env, make_env
from src.evaluate import evaluate_agent
from src.replay_buffer import ReplayBuffer
from src.utils import set_global_seed

from src.train_monitor import TrainingMonitor, TrainingSnapshot

def epsilon_by_step(step: int, config: BaseConfig) -> float:
    if getattr(config, "always_greedy_training", False):
        return 0.0
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def save_checkpoint(path: Path, agent: DQNAgent, config: BaseConfig, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online_net": agent.online_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "config": asdict(config),
            "summary": summary,
        },
        path,
    )


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_standard_dqn(config: BaseConfig) -> tuple[list[dict], list[dict], dict, DQNAgent]:
    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config.env_id, config.seed, sparse_reward=config.sparse_reward)
    eval_env = make_env(config.env_id, config.eval_seed, sparse_reward=config.sparse_reward)
    metadata = describe_env(env)
    agent = DQNAgent(
        state_dim=metadata["state_dim"],
        n_actions=metadata["n_actions"],
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gradient_clip_norm=config.gradient_clip_norm,
        seed=config.seed,
        device=device,
    )
    buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)
    mean_q_value = None
    mean_target = None
    episode_rows = []
    eval_rows = []
    state, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    gradient_steps = 0
    latest_loss = None

    wandb_run = None
    if getattr(config, "use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                group=config.experiment_name,
                name=f"{config.experiment_name}_seed_{config.seed}",
                config=asdict(config),
            )
            wandb.define_metric("env_step")
            wandb.define_metric("train/*", step_metric="env_step")
            wandb.define_metric("eval/*", step_metric="env_step")
        except ImportError:
            print("[WARNING] wandb is not installed. Run `pip install wandb` to enable cloud tracking.")

    with TrainingMonitor(
        total_env_steps=config.total_env_steps,
        experiment_name=config.experiment_name,
        device=str(device),
        update_every_steps=50,
    ) as monitor:
        for env_step in range(1, config.total_env_steps + 1):
            epsilon = epsilon_by_step(env_step - 1, config)
            selection = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(selection.action)
            buffer.push(
                state, selection.action, reward, next_state, terminated, truncated
            )
            state = next_state
            episode_return += reward
            episode_length += 1

            if len(buffer) >= config.learning_starts:
                batch = buffer.sample(config.batch_size)
                metrics = agent.gradient_update(batch)
                gradient_steps += 1
                latest_loss = metrics.loss
                mean_q_value = metrics.mean_q_value
                mean_target = metrics.mean_target

            if env_step % config.target_sync_interval == 0:
                agent.sync_target_network()

            # 周期性纯贪婪评测 (每 eval_interval 步执行一次)
            if config.eval_interval > 0 and env_step % config.eval_interval == 0:
                _, eval_metrics = evaluate_agent(
                    agent=agent,
                    env=eval_env,
                    episodes=config.eval_episodes,
                    evaluation_seed=config.eval_seed,
                )
                eval_record = {"env_step": env_step, **eval_metrics}
                eval_rows.append(eval_record)
                if wandb_run:
                    wandb.log({
                        "env_step": env_step,
                        "eval/mean_return": eval_metrics["mean_return"],
                        "eval/success_rate": eval_metrics["success_rate"],
                        "eval/mean_episode_length": eval_metrics["mean_episode_length"],
                    })

            if terminated or truncated:
                episode_index += 1
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "env_step": env_step,
                        "return": episode_return,
                        "length": episode_length,
                        "success": int(terminated),
                        "epsilon": epsilon,
                        "gradient_steps_so_far": gradient_steps,
                        "latest_loss": latest_loss,
                    }
                )
                if wandb_run:
                    wandb.log({
                        "env_step": env_step,
                        "train/episode_return": episode_return,
                        "train/episode_length": episode_length,
                        "train/epsilon": epsilon,
                        "train/loss": latest_loss if latest_loss is not None else 0.0,
                    })

                state, _ = env.reset(seed=config.seed + episode_index)
                monitor.update(
                    TrainingSnapshot(
                        env_step=env_step,
                        total_env_steps=config.total_env_steps,
                        episode=episode_index + 1,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        epsilon=epsilon,
                        replay_size=len(buffer),
                        replay_capacity=config.replay_capacity,
                        gradient_steps=gradient_steps,
                        latest_loss=latest_loss,
                        mean_q_value=mean_q_value,
                        mean_target=mean_target,
                    )
                )
                episode_return = 0.0
                episode_length = 0
                
    env.close()
    eval_env.close()
    if wandb_run:
        wandb.finish()

    summary = {
        "method": config.experiment_name,
        "seed": config.seed,
        "total_env_steps": config.total_env_steps,
        "completed_episodes": episode_index,
        "gradient_steps": gradient_steps,
        "final_epsilon": epsilon_by_step(config.total_env_steps - 1, config),
        "device": str(device),
        **asdict(config),
    }
    return episode_rows, eval_rows, summary, agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--sparse-reward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to use sparse goal reward (0 everywhere, +1 at goal)",
    )
    parser.add_argument(
        "--always-greedy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to always use pure greedy policy (epsilon=0.0) during training",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Step interval for periodic greedy evaluation (default: 10000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Number of episodes per evaluation checkpoint (default: 10)",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Weights & Biases experiment tracking",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom directory to save results (defaults to results/<experiment_name>/seed_<seed>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaseConfig()
    if args.total_env_steps is not None:
        config = replace(config, total_env_steps=args.total_env_steps)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.name is not None:
        config = replace(config, experiment_name=args.name)
    if args.sparse_reward is not None:
        config = replace(config, sparse_reward=args.sparse_reward)
    if args.always_greedy is not None:
        config = replace(config, always_greedy_training=args.always_greedy)
    if args.eval_interval is not None:
        config = replace(config, eval_interval=args.eval_interval)
    if args.eval_episodes is not None:
        config = replace(config, eval_episodes=args.eval_episodes)
    if args.use_wandb is not None:
        config = replace(config, use_wandb=args.use_wandb)
    if args.wandb_project is not None:
        config = replace(config, wandb_project=args.wandb_project)

    episode_rows, eval_rows, summary, agent = train_standard_dqn(config)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("results") / config.experiment_name / f"seed_{config.seed}"
    )
    save_rows(output_dir / "episodes.csv", episode_rows)
    save_rows(output_dir / "eval_during_train.csv", eval_rows)
    save_rows(output_dir / "summary.csv", [summary])
    save_checkpoint(output_dir / "checkpoint.pt", agent, config, summary)

    print("Standard DQN training completed")
    print("Output directory:", output_dir)
    print("Completed episodes:", summary["completed_episodes"])
    print("Gradient steps:", summary["gradient_steps"])
    if episode_rows:
        print("Final episode return:", episode_rows[-1]["return"])
        print("Final episode success:", episode_rows[-1]["success"])
    if eval_rows:
        print("Final greedy eval return:", eval_rows[-1]["mean_return"])


if __name__ == "__main__":
    main()
