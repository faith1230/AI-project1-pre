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
