from pathlib import Path

path = Path("src/train_dynamic.py")
if not path.exists():
    raise FileNotFoundError("Run this script from dynamic-dqn-project/.")

text = path.read_text(encoding="utf-8")
text = text.replace(
    '    save_rows(output_dir / "episodes.csv", episode_rows)\n',
    '    save_rows(output_dir / "episodes.csv", episode_rows)\n    checkpoint = {\n        "online_net": agent.online_net.state_dict(),\n        "target_net": agent.target_net.state_dict(),\n        "optimizer": agent.optimizer.state_dict(),\n        "env_step": config.total_env_steps,\n        "gradient_steps": summary["gradient_steps"],\n        "seed": config.seed,\n        "config": asdict(config),\n    }\n    output_dir.mkdir(parents=True, exist_ok=True)\n    torch.save(checkpoint, output_dir / "checkpoint_final.pt")\n'
)
path.write_text(text, encoding="utf-8")
print("Final checkpoint saving added to src/train_dynamic.py.")
