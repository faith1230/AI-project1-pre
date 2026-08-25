from pathlib import Path
from textwrap import dedent

ROOT = Path(".")

train_path = ROOT / "src/train.py"
fixed_path = ROOT / "src/train_fixed_frequency.py"
if not train_path.exists() or not fixed_path.exists():
    raise FileNotFoundError("Run this script from dynamic-dqn-project/ after Phase 5 and Phase 6.")

for path in (train_path, fixed_path):
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'from pathlib import Path\n\n        import torch',
        'from pathlib import Path\n\n        import torch'
    )
    if 'def save_checkpoint(' not in text:
        marker = '\n\n        def save_rows(path: Path, rows: list[dict]) -> None:'
        if marker not in text:
            marker = '\n\ndef save_rows(path: Path, rows: list[dict]) -> None:'
        function = dedent('''\


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
''')
        text = text.replace(marker, function + marker, 1)
    if 'save_checkpoint(output_dir / "checkpoint.pt"' not in text:
        needle = 'save_rows(output_dir / "summary.csv", [summary])'
        replacement = needle + '\n            save_checkpoint(output_dir / "checkpoint.pt", agent, config, summary)'
        text = text.replace(replacement if replacement in text else needle, replacement, 1)
    path.write_text(text, encoding="utf-8")

(ROOT / "src/evaluate.py").write_text(dedent('''\
    import argparse
    import csv
    from pathlib import Path

    import numpy as np
    import torch

    from configs.base_config import BaseConfig
    from src.dqn_agent import DQNAgent
    from src.environment import describe_env, make_env


    def load_agent(checkpoint_path: Path, env, seed: int) -> DQNAgent:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_config = checkpoint["config"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata = describe_env(env)
        agent = DQNAgent(
            state_dim=metadata["state_dim"],
            n_actions=metadata["n_actions"],
            hidden_dim=saved_config["hidden_dim"],
            learning_rate=saved_config["learning_rate"],
            gamma=saved_config["gamma"],
            gradient_clip_norm=saved_config["gradient_clip_norm"],
            seed=seed,
            device=device,
        )
        agent.online_net.load_state_dict(checkpoint["online_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.online_net.eval()
        agent.target_net.eval()
        return agent


    def evaluate(checkpoint_path: Path, episodes: int, seed: int) -> list[dict]:
        env = make_env("MountainCar-v0", seed)
        agent = load_agent(checkpoint_path, env, seed)
        rows = []
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                selection = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(selection.action)
                total_reward += reward
                length += 1
            rows.append({
                "episode": episode + 1,
                "return": total_reward,
                "length": length,
                "success": int(terminated),
                "terminated": int(terminated),
                "truncated": int(truncated),
            })
        env.close()
        return rows


    def save_rows(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--episodes", type=int, default=20)
        parser.add_argument("--seed", type=int, default=10_000)
        args = parser.parse_args()

        rows = evaluate(args.checkpoint, args.episodes, args.seed)
        output_path = args.checkpoint.parent / "evaluation.csv"
        save_rows(output_path, rows)
        returns = np.asarray([row["return"] for row in rows], dtype=np.float32)
        successes = np.asarray([row["success"] for row in rows], dtype=np.float32)
        print("Evaluation completed")
        print("Checkpoint:", args.checkpoint)
        print("Episodes:", len(rows))
        print("Mean return:", float(returns.mean()))
        print("Std return:", float(returns.std(ddof=1)) if len(rows) > 1 else 0.0)
        print("Success rate:", float(successes.mean()))
        print("Evaluation CSV:", output_path)


    if __name__ == "__main__":
        main()
'''), encoding="utf-8")

(ROOT / "src/compare_evaluation.py").write_text(dedent('''\
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
'''), encoding="utf-8")

print("Phase 9 checkpointing and evaluation files created successfully.")
