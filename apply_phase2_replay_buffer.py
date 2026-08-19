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
        """),
    "src/replay_buffer.py": dedent("""\
        from collections import deque
        from dataclasses import dataclass
        import random

        import numpy as np


        @dataclass(frozen=True)
        class Transition:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            terminated: bool
            truncated: bool

            @property
            def episode_done(self) -> bool:
                return self.terminated or self.truncated


        class ReplayBuffer:
            def __init__(self, capacity: int, seed: int) -> None:
                if capacity <= 0:
                    raise ValueError("capacity must be positive")
                self.memory = deque(maxlen=capacity)
                self.rng = random.Random(seed)

            def push(
                self,
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                terminated: bool,
                truncated: bool,
            ) -> None:
                transition = Transition(
                    state=np.asarray(state, dtype=np.float32).copy(),
                    action=int(action),
                    reward=float(reward),
                    next_state=np.asarray(next_state, dtype=np.float32).copy(),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                self.memory.append(transition)

            def sample(self, batch_size: int) -> list[Transition]:
                if not self.can_sample(batch_size):
                    raise ValueError(
                        f"Cannot sample {batch_size} transitions from a buffer of {len(self)}."
                    )
                return self.rng.sample(self.memory, batch_size)

            def can_sample(self, batch_size: int) -> bool:
                return len(self) >= batch_size

            def __len__(self) -> int:
                return len(self.memory)
        """),
    "src/smoke_test_replay.py": dedent("""\
        import numpy as np

        from configs.base_config import BaseConfig
        from src.replay_buffer import ReplayBuffer


        def main() -> None:
            config = BaseConfig()
            buffer = ReplayBuffer(capacity=32, seed=config.seed)

            for step in range(40):
                state = np.array([step, step + 0.1], dtype=np.float32)
                next_state = state + 1.0
                buffer.push(
                    state=state,
                    action=step % 3,
                    reward=-1.0,
                    next_state=next_state,
                    terminated=(step == 19),
                    truncated=(step == 39),
                )

            batch = buffer.sample(batch_size=8)
            assert len(buffer) == 32
            assert len(batch) == 8
            assert all(item.state.shape == (2,) for item in batch)
            assert all(isinstance(item.action, int) for item in batch)
            assert any(item.truncated for item in buffer.memory)

            print("Replay buffer smoke test passed")
            print("Current buffer size:", len(buffer))
            print("Sample batch size:", len(batch))
            print("One sampled transition:", batch[0])


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

print("Phase 2 replay-buffer files created successfully.")
