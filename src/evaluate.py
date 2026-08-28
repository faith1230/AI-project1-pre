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
