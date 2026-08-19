from pathlib import Path
from textwrap import dedent

ROOT = Path(".")

FILES = {
    "configs/base_config.py": dedent("""\
        from dataclasses import dataclass


        @dataclass(frozen=True)
        class BaseConfig:
            env_id: str = "MountainCar-v0"
            seed: int = 42
            experiment_name: str = "standard_dqn_baseline"
            total_env_steps: int = 100_000

            replay_capacity: int = 50_000
            batch_size: int = 64
            learning_starts: int = 1_000

            hidden_dim: int = 128
            gamma: float = 0.99
            learning_rate: float = 1e-3
            gradient_clip_norm: float = 10.0
            target_sync_interval: int = 1_000

            epsilon_start: float = 1.0
            epsilon_end: float = 0.05
            epsilon_decay_steps: int = 30_000
        """),
    "src/train.py": dedent("""\
        import argparse
        import csv
        from dataclasses import asdict, replace
        from pathlib import Path

        import torch

        from configs.base_config import BaseConfig
        from src.dqn_agent import DQNAgent
        from src.environment import describe_env, make_env
        from src.replay_buffer import ReplayBuffer
        from src.utils import set_global_seed


        def epsilon_by_step(step: int, config: BaseConfig) -> float:
            fraction = min(step / config.epsilon_decay_steps, 1.0)
            return config.epsilon_start + fraction * (
                config.epsilon_end - config.epsilon_start
            )


        def save_rows(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not rows:
                return
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


        def train_standard_dqn(config: BaseConfig) -> tuple[list[dict], dict]:
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
                    batch = buffer.sample(config.batch_size)
                    metrics = agent.gradient_update(batch)
                    gradient_steps += 1
                    latest_loss = metrics.loss

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
                "method": config.experiment_name,
                "seed": config.seed,
                "total_env_steps": config.total_env_steps,
                "completed_episodes": episode_index,
                "gradient_steps": gradient_steps,
                "final_epsilon": epsilon_by_step(config.total_env_steps - 1, config),
                "device": str(device),
                **asdict(config),
            }
            return episode_rows, summary


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

            episode_rows, summary = train_standard_dqn(config)
            output_dir = Path("results") / config.experiment_name / f"seed_{config.seed}"
            save_rows(output_dir / "episodes.csv", episode_rows)
            save_rows(output_dir / "summary.csv", [summary])

            print("Standard DQN training completed")
            print("Output directory:", output_dir)
            print("Completed episodes:", summary["completed_episodes"])
            print("Gradient steps:", summary["gradient_steps"])
            if episode_rows:
                print("Final episode return:", episode_rows[-1]["return"])
                print("Final episode success:", episode_rows[-1]["success"])


        if __name__ == "__main__":
            main()
        """),
}

for relative_path, content in FILES.items():
    path = ROOT / relative_path
    if not path.parent.exists():
        raise FileNotFoundError(
            f"Expected project folder '{path.parent}'. Run this script from dynamic-dqn-project/."
        )
    path.write_text(content, encoding="utf-8")

print("Phase 5 standard-DQN training files created successfully.")
