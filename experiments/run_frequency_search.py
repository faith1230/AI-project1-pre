import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch frequency search experiments comparing Fixed Frequency vs Dynamic DQN."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="List of training seeds (e.g. --seeds 42 43 44 45 46)",
    )
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 12, 16, 32],
        help="List of fixed update intervals to test (e.g. --intervals 1 2 4 8 16 32)",
    )
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=100_000,
        help="Total environment interaction steps per training run (default: 100000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per trained model (default: 100)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10_000,
        help="Starting seed for deterministic evaluation (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frequency_search_comparison.csv"),
        help="Output path for final comparison CSV",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun even if checkpoint/evaluation already exists",
    )
    args = parser.parse_args()

    python_exe = sys.executable

    tasks: list[tuple[str, str, list[str]]] = []

    # 1. Dynamic condition model
    dynamic_name = "dynamic_condition"
    tasks.append(("dynamic", dynamic_name, []))

    # 2. Fixed frequency models
    for interval in args.intervals:
        fixed_name = f"fixed_freq_{interval}"
        tasks.append(("fixed", fixed_name, ["--interval", str(interval)]))

    total_runs = len(tasks) * len(args.seeds)
    current_run = 0

    print("=" * 70)
    print(" BATCH FREQUENCY SEARCH EXPERIMENT")
    print(f" Seeds ({len(args.seeds)}): {args.seeds}")
    print(f" Intervals ({len(args.intervals)}): {args.intervals}")
    print(f" Total Env Steps: {args.total_env_steps}")
    print(f" Total Model Runs: {total_runs}")
    print("=" * 70)

    result_dirs: list[Path] = []

    for model_type, exp_name, extra_train_args in tasks:
        exp_dir = Path("results") / exp_name
        result_dirs.append(exp_dir)

        for seed in args.seeds:
            current_run += 1
            seed_dir = exp_dir / f"seed_{seed}"
            checkpoint_file = seed_dir / "checkpoint.pt"
            eval_summary_file = seed_dir / "evaluation" / "evaluation_summary.csv"

            print(f"\n[{current_run}/{total_runs}] >>> Method: {exp_name} | Seed: {seed}")

            # Step A: Train
            if (not args.force_rerun) and checkpoint_file.exists():
                print(f"[SKIP TRAIN] Checkpoint already exists: {checkpoint_file}")
            else:
                train_module = (
                    "src.train_dynamic" if model_type == "dynamic" else "src.train_fixed_frequency"
                )
                train_cmd = [
                    python_exe,
                    "-m",
                    train_module,
                    "--total-env-steps",
                    str(args.total_env_steps),
                    "--seed",
                    str(seed),
                    "--name",
                    exp_name,
                ] + extra_train_args
                run_command(train_cmd)

            # Step B: Evaluate
            if (not args.force_rerun) and eval_summary_file.exists():
                print(f"[SKIP EVAL] Evaluation summary already exists: {eval_summary_file}")
            else:
                eval_cmd = [
                    python_exe,
                    "-m",
                    "src.evaluate",
                    "--checkpoint",
                    str(checkpoint_file),
                    "--episodes",
                    str(args.eval_episodes),
                    "--evaluation-seed",
                    str(args.eval_seed),
                ]
                run_command(eval_cmd)

    # Step C: Compare all evaluations
    print("\n" + "=" * 70)
    print(" GENERATING FINAL COMPARISON")
    print("=" * 70)
    compare_cmd = [
        python_exe,
        "-m",
        "src.compare_evaluations",
        "--result-dirs",
        *[str(d) for d in result_dirs],
        "--output",
        str(args.output),
    ]
    run_command(compare_cmd)

    print("\n" + "=" * 70)
    print(f" ALL EXPERIMENTS COMPLETED!")
    print(f" Comparison table saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
