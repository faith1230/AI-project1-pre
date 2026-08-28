import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import torch

from configs.base_config import BaseConfig
from src.dqn_agent import DQNAgent
from src.environment import describe_env, make_env
from src.replay_buffer import ReplayBuffer
from src.train import epsilon_by_step
from src.utils import set_global_seed


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


def train_fixed_frequency(
    config: BaseConfig, update_interval: int
) -> tuple[list[dict], dict,DQNAgent]:
    if update_interval <= 0:
        raise ValueError("update_interval must be positive")

    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config.env_id, config.seed)
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

    episode_rows = []
    state, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    gradient_steps = 0
    steps_since_update = 0
    latest_loss = None

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
            steps_since_update += 1
            episode_done = terminated or truncated
            update_due = (
                steps_since_update >= update_interval or episode_done
            )
            if update_due:
                for _ in range(steps_since_update):
                    metrics = agent.gradient_update(
                        buffer.sample(config.batch_size)
                    )
                    gradient_steps += 1
                    latest_loss = metrics.loss
                steps_since_update = 0

        if env_step % config.target_sync_interval == 0:
            agent.sync_target_network()

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
            state, _ = env.reset(seed=config.seed + episode_index)
            episode_return = 0.0
            episode_length = 0

    env.close()
    summary = {
        "method": f"fixed_frequency_{update_interval}",
        "seed": config.seed,
        "total_env_steps": config.total_env_steps,
        "update_interval": update_interval,
        "completed_episodes": episode_index,
        "gradient_steps": gradient_steps,
        "final_epsilon": epsilon_by_step(config.total_env_steps - 1, config),
        "device": str(device),
        **asdict(config),
    }
    return episode_rows, summary,agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
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

    episode_rows, summary ,agent= train_fixed_frequency(config, args.interval)
    default_name = f"fixed_frequency_{args.interval}"
    experiment_name = args.name or default_name
    output_dir = Path("results") / experiment_name / f"seed_{config.seed}"
    save_rows(output_dir / "episodes.csv", episode_rows)
    save_rows(output_dir / "summary.csv", [summary])
    save_checkpoint(output_dir / "checkpoint.pt", agent, config, summary)

    print("Fixed-frequency DQN training completed")
    print("Update interval:", args.interval)
    print("Output directory:", output_dir)
    print("Completed episodes:", summary["completed_episodes"])
    print("Gradient steps:", summary["gradient_steps"])
    if episode_rows:
        print("Final episode return:", episode_rows[-1]["return"])
        print("Final episode success:", episode_rows[-1]["success"])


if __name__ == "__main__":
    main()
