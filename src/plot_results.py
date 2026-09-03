import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_comparison_metrics(comparison_path: Path, output_dir: Path):
    """Plot evaluation comparison metrics from results/comparison.csv."""
    df = pd.read_csv(comparison_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    metrics = [
        ("mean_return", "Mean Return", "Higher is better", "#2b5c8f"),
        ("success_rate", "Success Rate", "Higher is better", "#2a9d8f"),
        ("mean_episode_length", "Mean Episode Length", "Lower is better", "#e76f51"),
    ]

    for ax, (metric_base, title, note, color) in zip(axes, metrics):
        mean_col = f"{metric_base}_mean"
        ci_low_col = f"{metric_base}_ci95_low"
        ci_high_col = f"{metric_base}_ci95_high"

        methods = df["method"]
        means = df[mean_col]
        
        # Calculate asymmetric error bar lengths for CI95
        if ci_low_col in df.columns and ci_high_col in df.columns:
            yerr_low = np.maximum(0, means - df[ci_low_col])
            yerr_high = np.maximum(0, df[ci_high_col] - means)
            yerr = np.array([yerr_low, yerr_high])
        else:
            std_col = f"{metric_base}_std"
            yerr = df[std_col] if std_col in df.columns else None

        bars = ax.bar(methods, means, yerr=yerr, capsize=5, color=color, alpha=0.85, edgecolor="black", linewidth=1.2)
        
        for bar, m_val in zip(bars, means):
            y_pos = bar.get_height()
            offset = (abs(y_pos) * 0.04) if y_pos != 0 else 0.02
            if y_pos < 0:
                text_y = y_pos - offset
                va = "top"
            else:
                text_y = y_pos + offset
                va = "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{m_val:.2f}",
                ha="center",
                va=va,
                fontsize=10,
                fontweight="bold"
            )

        ax.set_title(f"{title}\n({note})", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Method", fontsize=11, fontweight="bold")
        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Model Evaluation Comparison on MountainCar-v0 (from comparison.csv)", fontsize=16, fontweight="bold", y=1.03)
    save_path = output_dir / "comparison_summary_dashboard.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(results_dir: Path, output_dir: Path, window: int = 10):
    """Plot smoothed training curves across all methods and seeds."""
    episode_files = list(results_dir.glob("*/seed_*/episodes.csv"))
    if not episode_files:
        print("No episode.csv files found for training curves.")
        return

    records = []
    for f in episode_files:
        method = f.parent.parent.name
        seed = f.parent.name
        df = pd.read_csv(f)
        df["method"] = method
        df["seed"] = seed
        if "return" in df.columns:
            df["smoothed_return"] = df["return"].rolling(window=window, min_periods=1).mean()
        records.append(df)

    all_df = pd.concat(records, ignore_index=True)
    if "episode" not in all_df.columns:
        all_df["episode"] = all_df.groupby(["method", "seed"]).cumcount() + 1

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.lineplot(
        data=all_df,
        x="episode",
        y="smoothed_return",
        hue="method",
        style="method",
        palette="tab10",
        errorbar=("ci", 95),
        linewidth=2.0,
        ax=ax,
    )
    ax.set_title(f"Training Learning Curves (Smoothed Window = {window})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax.set_ylabel("Smoothed Return", fontsize=12, fontweight="bold")
    ax.legend(title="Method", frameon=True, loc="lower right")
    plt.tight_layout()

    save_path = output_dir / "training_learning_curves.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_seed_distributions(results_dir: Path, output_dir: Path):
    """Plot evaluation score distribution across individual seeds."""
    summary_files = list(results_dir.glob("*/seed_*/evaluation/evaluation_summary.csv"))
    if not summary_files:
        print("No evaluation_summary.csv found.")
        return

    records = []
    for f in summary_files:
        method = f.parent.parent.parent.name
        seed = f.parent.parent.name
        df = pd.read_csv(f)
        df["method"] = method
        df["seed"] = seed
        records.append(df)

    eval_df = pd.concat(records, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Return distribution
    sns.boxplot(data=eval_df, x="method", y="mean_return", ax=axes[0], palette="Set2", boxprops=dict(alpha=0.7))
    sns.stripplot(data=eval_df, x="method", y="mean_return", ax=axes[0], color="black", size=7, jitter=0.2)
    axes[0].set_title("Seed-level Evaluation Return Distribution", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)

    # Success rate distribution
    sns.boxplot(data=eval_df, x="method", y="success_rate", ax=axes[1], palette="Set2", boxprops=dict(alpha=0.7))
    sns.stripplot(data=eval_df, x="method", y="success_rate", ax=axes[1], color="black", size=7, jitter=0.2)
    axes[1].set_title("Seed-level Success Rate Distribution", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    save_path = output_dir / "seed_level_distributions.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.comparison.exists():
        plot_comparison_metrics(args.comparison, args.output_dir)
    else:
        print(f"Warning: {args.comparison} does not exist.")

    plot_training_curves(args.results_dir, args.output_dir)
    plot_seed_distributions(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
