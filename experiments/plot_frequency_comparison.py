import argparse
import csv
from pathlib import Path

def plot_results(csv_path: Path, output_image: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib is not installed. Run `pip install matplotlib` to generate plots.")
        return

    with csv_path.open(encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    dynamic_row = None
    fixed_rows = []

    for row in reader:
        method = row["method"]
        if "dynamic" in method:
            dynamic_row = row
        elif "fixed" in method:
            # Extract interval number from method name (e.g. fixed_freq_4 -> 4)
            parts = method.split("_")
            interval = int(parts[-1]) if parts[-1].isdigit() else 1
            fixed_rows.append((interval, row))

    fixed_rows.sort(key=lambda x: x[0])

    if not fixed_rows:
        print("[ERROR] No fixed frequency rows found in comparison CSV.")
        return

    intervals = [item[0] for item in fixed_rows]
    mean_returns = [float(item[1]["mean_return_mean"]) for item in fixed_rows]
    ci_lows = [float(item[1]["mean_return_ci95_low"]) for item in fixed_rows]
    ci_highs = [float(item[1]["mean_return_ci95_high"]) for item in fixed_rows]
    yerr = [
        [mean - low for mean, low in zip(mean_returns, ci_lows)],
        [high - mean for mean, high in zip(mean_returns, ci_highs)],
    ]

    success_rates = [float(item[1]["success_rate_mean"]) for item in fixed_rows]
    sr_ci_lows = [float(item[1]["success_rate_ci95_low"]) for item in fixed_rows]
    sr_ci_highs = [float(item[1]["success_rate_ci95_high"]) for item in fixed_rows]
    sr_yerr = [
        [sr - low for sr, low in zip(success_rates, sr_ci_lows)],
        [high - sr for sr, high in zip(success_rates, sr_ci_highs)],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Mean Return Plot
    ax1.errorbar(
        intervals,
        mean_returns,
        yerr=yerr,
        fmt="o-",
        capsize=4,
        color="#1f77b4",
        label="Fixed Frequency (95% CI)",
        linewidth=2,
    )
    if dynamic_row:
        dyn_mean = float(dynamic_row["mean_return_mean"])
        dyn_low = float(dynamic_row["mean_return_ci95_low"])
        dyn_high = float(dynamic_row["mean_return_ci95_high"])
        ax1.axhline(dyn_mean, color="#d62728", linestyle="--", label=f"Dynamic DQN Mean ({dyn_mean:.1f})")
        ax1.axhspan(dyn_low, dyn_high, color="#d62728", alpha=0.15, label="Dynamic 95% CI")

    ax1.set_xlabel("Update Interval (Fixed Frequency)")
    ax1.set_ylabel("Evaluation Mean Return")
    ax1.set_title("Return vs. Update Interval")
    ax1.set_xticks(intervals)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Success Rate Plot
    ax2.errorbar(
        intervals,
        success_rates,
        yerr=sr_yerr,
        fmt="s-",
        capsize=4,
        color="#2ca02c",
        label="Fixed Frequency (95% CI)",
        linewidth=2,
    )
    if dynamic_row:
        dyn_sr = float(dynamic_row["success_rate_mean"])
        dyn_sr_low = float(dynamic_row["success_rate_ci95_low"])
        dyn_sr_high = float(dynamic_row["success_rate_ci95_high"])
        ax2.axhline(dyn_sr, color="#d62728", linestyle="--", label=f"Dynamic DQN Mean ({dyn_sr:.2f})")
        ax2.axhspan(dyn_sr_low, dyn_sr_high, color="#d62728", alpha=0.15, label="Dynamic 95% CI")

    ax2.set_xlabel("Update Interval (Fixed Frequency)")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rate vs. Update Interval")
    ax2.set_xticks(intervals)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[SUCCESS] Plot saved to: {output_image}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frequency comparison results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/frequency_search_comparison.csv"),
        help="Path to comparison CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frequency_comparison_plot.png"),
        help="Path to save plot image",
    )
    args = parser.parse_args()
    plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
