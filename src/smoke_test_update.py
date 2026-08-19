import torch

from configs.base_config import BaseConfig
from src.dqn_agent import DQNAgent
from src.environment import describe_env, make_env
from src.replay_buffer import ReplayBuffer
from src.utils import set_global_seed


def main() -> None:
    config = BaseConfig()
    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config.env_id, config.seed)
    metadata = describe_env(env)
    buffer = ReplayBuffer(capacity=128, seed=config.seed)

    state, _ = env.reset(seed=config.seed)
    for step in range(80):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, terminated, truncated)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset(seed=config.seed + step + 1)

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

    online_before = [p.detach().clone() for p in agent.online_net.parameters()]
    target_before = [p.detach().clone() for p in agent.target_net.parameters()]
    metrics = agent.gradient_update(buffer.sample(config.batch_size))

    online_changed = any(
        not torch.equal(before, after)
        for before, after in zip(online_before, agent.online_net.parameters())
    )
    target_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(target_before, agent.target_net.parameters())
    )
    assert online_changed
    assert target_unchanged
    assert metrics.loss >= 0.0

    agent.sync_target_network()
    target_matches_online = all(
        torch.equal(online, target)
        for online, target in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        )
    )
    assert target_matches_online

    print("DQN gradient-update smoke test passed")
    print("Device:", device)
    print("Loss:", metrics.loss)
    print("Mean predicted Q:", metrics.mean_q_value)
    print("Mean Bellman target:", metrics.mean_target)
    env.close()


if __name__ == "__main__":
    main()
