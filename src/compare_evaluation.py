import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_csv", nargs="+", type=Path)
    args = parser.parse_args()

    output_rows = []
    for path in args.evaluation_csv:
        rows = read_rows(path)
        returns = np.asarray([float(row["return"]) for row in rows])
        successes = np.asarray([int(row["success"]) for row in rows])
        output_rows.append({
            "method": path.parent.parent.name,
            "seed": path.parent.name,
            "episodes": len(rows),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std(ddof=1)) if len(rows) > 1 else 0.0,
            "success_rate": float(successes.mean()),
        })

    print("method,seed,episodes,mean_return,std_return,success_rate")
    for row in output_rows:
        print(
            f'{row["method"]},{row["seed"]},{row["episodes"]},'
            f'{row["mean_return"]:.3f},{row["std_return"]:.3f},'
            f'{row["success_rate"]:.3f}'
        )
