from pathlib import Path
from textwrap import dedent

ROOT = Path(".")

FILES = {
    "configs/base_config.py": dedent("""\
        from dataclasses import dataclass


        @dataclass(frozen=True)
        class BaseConfig:
            env_id: str = "MountainCar-v0"
            seed: int = 42
            max_episode_steps: int = 200
            experiment_name: str = "standard_dqn_baseline"

            replay_capacity: int = 50_000
            batch_size: int = 64
            learning_starts: int = 1_000

            hidden_dim: int = 128
            gamma: float = 0.99
            learning_rate: float = 1e-3
            epsilon_start: float = 1.0
            epsilon_end: float = 0.05
            epsilon_decay_steps: int = 30_000
        """),
    "src/dqn_agent.py": dedent("""\
        from dataclasses import dataclass
        import random

        import numpy as np
        import torch
        from torch import nn


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


        class DQNAgent:
            def __init__(
                self,
                state_dim: int,
                n_actions: int,
                hidden_dim: int,
                seed: int,
                device: torch.device,
            ) -> None:
                self.n_actions = n_actions
                self.device = device
                self.rng = random.Random(seed)
                self.online_net = QNetwork(state_dim, n_actions, hidden_dim).to(device)

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
        """),
    "src/smoke_test_agent.py": dedent("""\
        import torch

        from configs.base_config import BaseConfig
        from src.dqn_agent import DQNAgent
        from src.environment import describe_env, make_env
        from src.utils import set_global_seed


        def main() -> None:
            config = BaseConfig()
            set_global_seed(config.seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            env = make_env(config.env_id, config.seed)
            metadata = describe_env(env)
            state, _ = env.reset(seed=config.seed)

            agent = DQNAgent(
                state_dim=metadata["state_dim"],
                n_actions=metadata["n_actions"],
                hidden_dim=config.hidden_dim,
                seed=config.seed,
                device=device,
            )

            q_values = agent.q_values(state)
            greedy_choice = agent.select_action(state, epsilon=0.0)
            exploratory_choice = agent.select_action(state, epsilon=1.0)

            assert q_values.shape == (metadata["n_actions"],)
            assert greedy_choice.is_greedy is True
            assert greedy_choice.action == int(torch.argmax(q_values).item())
            assert exploratory_choice.is_greedy is False
            assert 0 <= exploratory_choice.action < metadata["n_actions"]

            print("Q-network and action-selection smoke test passed")
            print("Device:", device)
            print("Q-values:", q_values.cpu().numpy())
            print("Greedy choice:", greedy_choice)
            print("Exploratory choice:", exploratory_choice)
            env.close()


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

print("Phase 3 Q-network files created successfully.")
