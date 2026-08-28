from pathlib import Path

import torch

from configs.base_config import BaseConfig
from src.dqn_agent import DQNAgent




def load_agent(path: Path, state_dim: int, n_actions: int, device: torch.device) -> DQNAgent:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = BaseConfig(**checkpoint["config"])
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gradient_clip_norm=config.gradient_clip_norm,
        seed=config.seed,
        device=device,
    )
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.online_net.eval()
    agent.target_net.eval()
    return agent
