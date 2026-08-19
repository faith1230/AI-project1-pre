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
