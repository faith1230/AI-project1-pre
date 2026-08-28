from dataclasses import dataclass


@dataclass(frozen=True)
class BaseConfig:
    env_id: str = "MountainCar-v0"
    seed: int = 42
    experiment_name: str = "standard_dqn_baseline"
    total_env_steps: int = 100_000

    replay_capacity: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000

    hidden_dim: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 10.0
    target_sync_interval: int = 100

    epsilon_start: float = 1.0
    epsilon_end: float = 0.005
    epsilon_decay_steps: int = 50_000
