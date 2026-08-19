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
