from pathlib import Path
from textwrap import dedent

ROOT = Path(".")

content = dedent("""\
    import argparse
    import csv
    from dataclasses import asdict, replace
    from pathlib import Path

    import numpy as np
    import torch

    from configs.base_config import BaseConfig
    from src.dqn_agent import DQNAgent
    from src.environment import describe_env, make_env
    from src.replay_buffer import ReplayBuffer
    from src.train import epsilon_by_step
    from src.utils import set_global_seed


    def dynamic_update_condition(
        last_value: float, reward: float, value: float
    ) -> bool:
        return bool(np.sign(value) * (last_value - (reward + value)) >= 0.0)


    def save_rows(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


    def train_dynamic_dqn(config: BaseConfig) -> tuple[list[dict], list[dict], dict]:
        set_global_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = make_env(config.env_id, config.seed)
        metadata = describe_env(env)
        state_dim = metadata["state_dim"]
        agent = DQNAgent(
            state_dim=state_dim,
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
        step_rows = []
        state, _ = env.reset(seed=config.seed)
        episode_return = 0.0
        episode_length = 0
        episode_index = 0
        gradient_steps = 0
        steps_since_update = 0
        latest_loss = None
        total_greedy_actions = 0
        total_exploratory_actions = 0
        total_condition_triggers = 0
        episode_greedy_actions = 0
        episode_exploratory_actions = 0
        episode_condition_triggers = 0

        for env_step in range(1, config.total_env_steps + 1):
            epsilon = epsilon_by_step(env_step - 1, config)
            selection = agent.select_action(state, epsilon)
            last_value = agent.state_value(state) if selection.is_greedy else None
            next_state, reward, terminated, truncated, _ = env.step(selection.action)
            buffer.push(
                state, selection.action, reward, next_state, terminated, truncated
            )
            episode_done = terminated or truncated

            if selection.is_greedy:
                total_greedy_actions += 1
                episode_greedy_actions += 1
            else:
                total_exploratory_actions += 1
                episode_exploratory_actions += 1

            current_value = None
            condition_evaluated = False
            condition_triggered = False
            update_due = False
            pending_updates_before = 0
            updates_this_interaction = 0

            if len(buffer) >= config.learning_starts:
                steps_since_update += 1
                pending_updates_before = steps_since_update

                if selection.is_greedy:
                    current_value = agent.state_value(next_state)
                    condition_evaluated = True
                    condition_triggered = dynamic_update_condition(
                        last_value, reward, current_value
                    )
                    if condition_triggered:
                        total_condition_triggers += 1
                        episode_condition_triggers += 1

                update_due = condition_triggered or episode_done
                if update_due:
                    for _ in range(steps_since_update):
                        metrics = agent.gradient_update(buffer.sample(config.batch_size))
                        gradient_steps += 1
                        updates_this_interaction += 1
                        latest_loss = metrics.loss
                    steps_since_update = 0

            target_synced = env_step % config.target_sync_interval == 0
            if target_synced:
                agent.sync_target_network()

            step_row = {
                "env_step": env_step,
                "episode": episode_index + 1,
                "step_in_episode": episode_length + 1,
                "epsilon": epsilon,
                "state_0": float(state[0]),
                "state_1": float(state[1]),
                "action": selection.action,
                "is_greedy": int(selection.is_greedy),
                "last_value": last_value,
                "reward": float(reward),
                "next_state_0": float(next_state[0]),
                "next_state_1": float(next_state[1]),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "episode_done": int(episode_done),
                "current_value": current_value,
                "condition_evaluated": int(condition_evaluated),
                "condition_triggered": int(condition_triggered),
                "pending_updates_before": pending_updates_before,
                "updates_this_interaction": updates_this_interaction,
                "gradient_steps_so_far": gradient_steps,
                "replay_buffer_size": len(buffer),
                "target_synced": int(target_synced),
                "latest_loss": latest_loss,
            }
            if state_dim != 2:
                raise ValueError("Step CSV columns currently expect MountainCar's 2-D state.")
            step_rows.append(step_row)

            state = next_state
            episode_return += reward
            episode_length += 1

            if episode_done:
                episode_index += 1
                episode_rows.append(
                    {
                        "episode": episode_index,
                        "env_step": env_step,
                        "return": episode_return,
                        "length": episode_length,
                        "success": int(terminated),
                        "epsilon": epsilon,
                        "greedy_actions": episode_greedy_actions,
                        "exploratory_actions": episode_exploratory_actions,
                        "condition_triggers": episode_condition_triggers,
                        "gradient_steps_so_far": gradient_steps,
                        "latest_loss": latest_loss,
                    }
                )
                state, _ = env.reset(seed=config.seed + episode_index)
                episode_return = 0.0
                episode_length = 0
                episode_greedy_actions = 0
                episode_exploratory_actions = 0
                episode_condition_triggers = 0

        env.close()
        summary = {
            "method": "dynamic_condition",
            "seed": config.seed,
            "total_env_steps": config.total_env_steps,
            "completed_episodes": episode_index,
            "gradient_steps": gradient_steps,
            "greedy_actions": total_greedy_actions,
            "exploratory_actions": total_exploratory_actions,
            "condition_triggers": total_condition_triggers,
            "condition_trigger_rate_among_greedy": (
                total_condition_triggers / total_greedy_actions
                if total_greedy_actions else 0.0
            ),
            "final_epsilon": epsilon_by_step(config.total_env_steps - 1, config),
            "device": str(device),
            **asdict(config),
        }
        return episode_rows, step_rows, summary


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
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

        episode_rows, step_rows, summary = train_dynamic_dqn(config)
        experiment_name = args.name or "dynamic_condition"
        output_dir = Path("results") / experiment_name / f"seed_{config.seed}"
        save_rows(output_dir / "episodes.csv", episode_rows)
        save_rows(output_dir / "steps.csv", step_rows)
        save_rows(output_dir / "summary.csv", [summary])

        print("Dynamic-condition DQN training completed")
        print("Output directory:", output_dir)
        print("Logged environment steps:", len(step_rows))
        print("Completed episodes:", summary["completed_episodes"])
        print("Gradient steps:", summary["gradient_steps"])
        print("Greedy actions:", summary["greedy_actions"])
        print("Exploratory actions:", summary["exploratory_actions"])
        print("Condition triggers:", summary["condition_triggers"])
        print("Condition-trigger rate among greedy actions:", summary["condition_trigger_rate_among_greedy"])


    if __name__ == "__main__":
        main()
    """)

path = ROOT / "src/train_dynamic.py"
if not path.parent.exists():
    raise FileNotFoundError(
        "Expected src/. Run this script from dynamic-dqn-project/."
    )
path.write_text(content, encoding="utf-8")
print("Phase 8 per-step logging added to src/train_dynamic.py.")
