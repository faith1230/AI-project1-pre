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
