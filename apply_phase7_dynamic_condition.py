from pathlib import Path
from textwrap import dedent

ROOT = Path(".")

FILES = {
    "src/dqn_agent.py": dedent("""\
        from copy import deepcopy
        from dataclasses import dataclass
        import random

        import numpy as np
        import torch
        from torch import nn

        from src.replay_buffer import Transition


        class QNetwork(nn.Module):
            def __init__(self, state_dim: int, n_actions: int, hidden_dim: int) -> None:
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_actions),
                )

            def forward(self, states: torch.Tensor) -> torch.Tensor:
                return self.model(states)


        @dataclass(frozen=True)
        class ActionSelection:
            action: int
            is_greedy: bool


        @dataclass(frozen=True)
        class UpdateMetrics:
            loss: float
            mean_q_value: float
            mean_target: float


        class DQNAgent:
            def __init__(
                self,
                state_dim: int,
                n_actions: int,
                hidden_dim: int,
                learning_rate: float,
                gamma: float,
                gradient_clip_norm: float,
                seed: int,
                device: torch.device,
            ) -> None:
                self.n_actions = n_actions
                self.gamma = gamma
                self.gradient_clip_norm = gradient_clip_norm
                self.device = device
                self.rng = random.Random(seed)

                self.online_net = QNetwork(state_dim, n_actions, hidden_dim).to(device)
                self.target_net = deepcopy(self.online_net).to(device)
                self.target_net.eval()
                for parameter in self.target_net.parameters():
                    parameter.requires_grad_(False)

                self.optimizer = torch.optim.Adam(
                    self.online_net.parameters(), lr=learning_rate
                )
                self.loss_fn = nn.SmoothL1Loss()

            def q_values(self, state: np.ndarray) -> torch.Tensor:
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                was_training = self.online_net.training
                self.online_net.eval()
                with torch.no_grad():
                    values = self.online_net(state_tensor).squeeze(0)
                self.online_net.train(was_training)
                return values

            def state_value(self, state: np.ndarray) -> float:
                return float(self.q_values(state).max().item())

            def select_action(self, state: np.ndarray, epsilon: float) -> ActionSelection:
                if not 0.0 <= epsilon <= 1.0:
                    raise ValueError("epsilon must be between 0 and 1")

                if self.rng.random() < epsilon:
                    return ActionSelection(
                        action=self.rng.randrange(self.n_actions),
                        is_greedy=False,
                    )

                values = self.q_values(state)
                return ActionSelection(
                    action=int(torch.argmax(values).item()),
                    is_greedy=True,
                )

            def gradient_update(self, batch: list[Transition]) -> UpdateMetrics:
                states = torch.as_tensor(
                    np.stack([item.state for item in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    [item.action for item in batch],
                    dtype=torch.int64,
                    device=self.device,
                ).unsqueeze(1)
                rewards = torch.as_tensor(
                    [item.reward for item in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                next_states = torch.as_tensor(
                    np.stack([item.next_state for item in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                terminated = torch.as_tensor(
                    [item.terminated for item in batch],
                    dtype=torch.float32,
                    device=self.device,
                )

                predicted_q = self.online_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q = self.target_net(next_states).max(dim=1).values
                    targets = rewards + self.gamma * (1.0 - terminated) * next_q

                loss = self.loss_fn(predicted_q, targets)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), max_norm=self.gradient_clip_norm
                )
                self.optimizer.step()

                return UpdateMetrics(
                    loss=float(loss.item()),
                    mean_q_value=float(predicted_q.detach().mean().item()),
                    mean_target=float(targets.mean().item()),
                )

            def sync_target_network(self) -> None:
                self.target_net.load_state_dict(self.online_net.state_dict())
        """),
    "src/train_dynamic.py": dedent("""\
        import argparse
        import csv
        from dataclasses import asdict, replace
        from pathlib import Path

        import numpy as np
        import torch

        from configs.base_config import BaseConfig
        from src.dqn_agent import DQNAgent
        from src.environment import describe_env, make_env
        from src.replay_buffer import ReplayBuffer
        from src.train import epsilon_by_step
        from src.utils import set_global_seed


        def dynamic_update_condition(
            last_value: float, reward: float, value: float
        ) -> bool:
            return bool(np.sign(value) * (last_value - (reward + value)) >= 0.0)


        def save_rows(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not rows:
                return
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


        def train_dynamic_dqn(config: BaseConfig) -> tuple[list[dict], dict]:
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
            steps_since_update = 0
            latest_loss = None
            total_greedy_actions = 0
            total_exploratory_actions = 0
            total_condition_triggers = 0
            episode_greedy_actions = 0
            episode_exploratory_actions = 0
            episode_condition_triggers = 0

            for env_step in range(1, config.total_env_steps + 1):
                epsilon = epsilon_by_step(env_step - 1, config)
                selection = agent.select_action(state, epsilon)
                last_value = agent.state_value(state) if selection.is_greedy else None
                next_state, reward, terminated, truncated, _ = env.step(selection.action)
                buffer.push(
                    state, selection.action, reward, next_state, terminated, truncated
                )
                episode_done = terminated or truncated

                if selection.is_greedy:
                    total_greedy_actions += 1
                    episode_greedy_actions += 1
                else:
                    total_exploratory_actions += 1
                    episode_exploratory_actions += 1

                condition_triggered = False
                if len(buffer) >= config.learning_starts:
                    steps_since_update += 1
                    if selection.is_greedy:
                        value = agent.state_value(next_state)
                        condition_triggered = dynamic_update_condition(
                            last_value, reward, value
                        )
                        if condition_triggered:
                            total_condition_triggers += 1
                            episode_condition_triggers += 1

                    update_due = condition_triggered or episode_done
                    if update_due:
                        for _ in range(steps_since_update):
                            metrics = agent.gradient_update(
                                buffer.sample(config.batch_size)
                            )
                            gradient_steps += 1
                            latest_loss = metrics.loss
                        steps_since_update = 0

                if env_step % config.target_sync_interval == 0:
                    agent.sync_target_network()

                state = next_state
                episode_return += reward
                episode_length += 1

                if episode_done:
                    episode_index += 1
                    episode_rows.append(
                        {
                            "episode": episode_index,
                            "env_step": env_step,
                            "return": episode_return,
                            "length": episode_length,
                            "success": int(terminated),
                            "epsilon": epsilon,
                            "greedy_actions": episode_greedy_actions,
                            "exploratory_actions": episode_exploratory_actions,
                            "condition_triggers": episode_condition_triggers,
                            "gradient_steps_so_far": gradient_steps,
                            "latest_loss": latest_loss,
                        }
                    )
                    state, _ = env.reset(seed=config.seed + episode_index)
                    episode_return = 0.0
                    episode_length = 0
                    episode_greedy_actions = 0
                    episode_exploratory_actions = 0
                    episode_condition_triggers = 0

            env.close()
            summary = {
                "method": "dynamic_condition",
                "seed": config.seed,
                "total_env_steps": config.total_env_steps,
                "completed_episodes": episode_index,
                "gradient_steps": gradient_steps,
                "greedy_actions": total_greedy_actions,
                "exploratory_actions": total_exploratory_actions,
                "condition_triggers": total_condition_triggers,
                "condition_trigger_rate_among_greedy": (
                    total_condition_triggers / total_greedy_actions
                    if total_greedy_actions else 0.0
                ),
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

            episode_rows, summary = train_dynamic_dqn(config)
            experiment_name = args.name or "dynamic_condition"
            output_dir = Path("results") / experiment_name / f"seed_{config.seed}"
            save_rows(output_dir / "episodes.csv", episode_rows)
            save_rows(output_dir / "summary.csv", [summary])

            print("Dynamic-condition DQN training completed")
            print("Output directory:", output_dir)
            print("Completed episodes:", summary["completed_episodes"])
            print("Gradient steps:", summary["gradient_steps"])
            print("Greedy actions:", summary["greedy_actions"])
            print("Exploratory actions:", summary["exploratory_actions"])
            print("Condition triggers:", summary["condition_triggers"])
            print("Condition-trigger rate among greedy actions:", summary["condition_trigger_rate_among_greedy"])


        if __name__ == "__main__":
            main()
        """),
    "src/smoke_test_dynamic_condition.py": dedent("""\
        from src.train_dynamic import dynamic_update_condition


        def main() -> None:
            assert dynamic_update_condition(5.0, 1.0, 2.0) is True
            assert dynamic_update_condition(1.0, 1.0, 2.0) is False
            assert dynamic_update_condition(-5.0, -1.0, -2.0) is True
            assert dynamic_update_condition(-1.0, -1.0, -2.0) is False
            assert dynamic_update_condition(10.0, 999.0, 0.0) is True
            print("Dynamic update-condition smoke test passed")


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

print("Phase 7 dynamic-condition files created successfully.")
