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
    sparse_reward: bool = False

    # 评估与贪婪策略配置
    always_greedy_training: bool = True   # 训练时是否全程使用纯贪婪策略 (epsilon=0.0)
    eval_interval: int = 10_000           # 每训练多少 steps 评估一次 (纯 greedy 策略)
    eval_episodes: int = 10               # 每次评估的回合数
    eval_seed: int = 10_000               # 评估基准随机种子

    # WandB 跟踪配置
    use_wandb: bool = True                # 是否启用 wandb.ai 实验跟踪
    wandb_project: str = "MountainCar-DQN"# wandb 项目名
