from pathlib import Path
from textwrap import dedent

ROOT = Path(".")

NEW_FILES = {
    "src/checkpoint.py": dedent("""\
        from dataclasses import asdict
        from pathlib import Path

        import torch

        from configs.base_config import BaseConfig
        from src.dqn_agent import DQNAgent


        def save_checkpoint(path: Path, agent: DQNAgent, config: BaseConfig) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config": asdict(config),
                    "online_net_state_dict": agent.online_net.state_dict(),
                    "target_net_state_dict": agent.target_net.state_dict(),
                },
                path,
            )


        def load_agent(path: Path, state_dim: int, n_actions: int, device: torch.device) -> DQNAgent:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            config = BaseConfig(**checkpoint["config"])
            agent = DQNAgent(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dim=config.hidden_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                gradient_clip_norm=config.gradient_clip_norm,
                seed=config.seed,
                device=device,
            )
            agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
            agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            agent.online_net.eval()
            agent.target_net.eval()
            return agent
        """),
    "src/evaluate.py": dedent("""\
        import argparse
        import csv
        from pathlib import Path

        import numpy as np
        import torch

        from src.checkpoint import load_agent
        from src.environment import describe_env, make_env


        def save_rows(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


        def evaluate_checkpoint(
            checkpoint_path: Path, episodes: int, evaluation_seed: int
        ) -> tuple[list[dict], dict]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            env = make_env("MountainCar-v0", evaluation_seed)
            metadata = describe_env(env)
            agent = load_agent(
                checkpoint_path,
                state_dim=metadata["state_dim"],
                n_actions=metadata["n_actions"],
                device=device,
            )

            rows = []
            for episode in range(episodes):
                state, _ = env.reset(seed=evaluation_seed + episode)
                episode_return = 0.0
                length = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action = agent.select_action(state, epsilon=0.0).action
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_return += reward
                    length += 1

                rows.append(
                    {
                        "evaluation_episode": episode + 1,
                        "evaluation_seed": evaluation_seed + episode,
                        "return": episode_return,
                        "length": length,
                        "success": int(terminated),
                    }
                )

            env.close()
            returns = np.asarray([row["return"] for row in rows], dtype=float)
            lengths = np.asarray([row["length"] for row in rows], dtype=float)
            successes = np.asarray([row["success"] for row in rows], dtype=float)
            successful_lengths = lengths[successes == 1]
            summary = {
                "checkpoint": str(checkpoint_path),
                "evaluation_episodes": episodes,
                "evaluation_seed_start": evaluation_seed,
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std(ddof=1)) if episodes > 1 else 0.0,
                "median_return": float(np.median(returns)),
                "success_rate": float(successes.mean()),
                "mean_episode_length": float(lengths.mean()),
                "mean_length_when_successful": (
                    float(successful_lengths.mean()) if len(successful_lengths) else float("nan")
                ),
            }
            return rows, summary


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint", type=Path, required=True)
            parser.add_argument("--episodes", type=int, default=100)
            parser.add_argument("--evaluation-seed", type=int, default=10_000)
            args = parser.parse_args()

            rows, summary = evaluate_checkpoint(
                args.checkpoint, args.episodes, args.evaluation_seed
            )
            output_dir = args.checkpoint.parent / "evaluation"
            save_rows(output_dir / "evaluation_episodes.csv", rows)
            save_rows(output_dir / "evaluation_summary.csv", [summary])
            print("Evaluation completed")
            for key, value in summary.items():
                print(f"{key}: {value}")


        if __name__ == "__main__":
            main()
        """),
    "src/compare_evaluations.py": dedent("""\
        import argparse
        import csv
        from pathlib import Path

        import numpy as np


        METRICS = ["mean_return", "success_rate", "mean_episode_length"]


        def read_summary(path: Path) -> dict:
            with path.open(encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            return row


        def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 10_000) -> tuple[float, float]:
            rng = np.random.default_rng(seed)
            samples = rng.choice(values, size=(repeats, len(values)), replace=True)
            means = samples.mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            return float(low), float(high)


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--result-dirs", type=Path, nargs="+", required=True)
            parser.add_argument("--output", type=Path, default=Path("results/comparison.csv"))
            args = parser.parse_args()

            rows = []
            for result_dir in args.result_dirs:
                summaries = sorted(result_dir.glob("seed_*/evaluation/evaluation_summary.csv"))
                if not summaries:
                    raise FileNotFoundError(
                        f"No evaluation summaries found under {result_dir}/seed_*/evaluation/"
                    )
                loaded = [read_summary(path) for path in summaries]
                row = {"method": result_dir.name, "n_seeds": len(loaded)}
                for metric_index, metric in enumerate(METRICS):
                    values = np.asarray([float(item[metric]) for item in loaded], dtype=float)
                    ci_low, ci_high = bootstrap_ci(values, seed=2026 + metric_index)
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    row[f"{metric}_ci95_low"] = ci_low
                    row[f"{metric}_ci95_high"] = ci_high
                rows.append(row)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print("Comparison written to:", args.output)
            for row in rows:
                print(row)


        if __name__ == "__main__":
            main()
        """),
}

for relative_path, content in NEW_FILES.items():
    path = ROOT / relative_path
    if not path.parent.exists():
        raise FileNotFoundError(
            f"Expected project folder '{path.parent}'. Run this script from dynamic-dqn-project/."
        )
    path.write_text(content, encoding="utf-8")

TRAINING_FILES = [
    ("src/train.py", "train_standard_dqn(config)"),
    ("src/train_fixed_frequency.py", "train_fixed_frequency(config, args.interval)"),
    ("src/train_dynamic.py", "train_dynamic_dqn(config)"),
]

for relative_path, function_call in TRAINING_FILES:
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    if "from src.checkpoint import save_checkpoint" not in text:
        text = text.replace(
            "from src.dqn_agent import DQNAgent\n",
            "from src.checkpoint import save_checkpoint\nfrom src.dqn_agent import DQNAgent\n",
        )
    text = text.replace(
        "return episode_rows, summary\n",
        "return episode_rows, summary, agent\n",
    )
    text = text.replace(
        f"episode_rows, summary = {function_call}\n",
        f"episode_rows, summary, agent = {function_call}\n",
    )
    marker = 'save_rows(output_dir / "episodes.csv", episode_rows)\n'
    replacement = (
        'save_checkpoint(output_dir / "model.pt", agent, config)\n'
        + marker
    )
    if marker in text and "save_checkpoint(output_dir / \"model.pt\"" not in text:
        text = text.replace(marker, replacement)
    if "return episode_rows, summary, agent" not in text:
        raise RuntimeError(f"Could not patch the return value in {relative_path}.")
    if "save_checkpoint(output_dir / \"model.pt\"" not in text:
        raise RuntimeError(f"Could not patch checkpoint saving in {relative_path}.")
    path.write_text(text, encoding="utf-8")

print("Phase 8 checkpointing and evaluation files created successfully.")
