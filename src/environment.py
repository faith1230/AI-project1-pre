import gymnasium as gym


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def describe_env(env) -> dict:
    if len(env.observation_space.shape) != 1:
        raise ValueError("This project currently expects a 1-D vector observation.")

    return {
        "state_dim": env.observation_space.shape[0],
        "n_actions": env.action_space.n,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }
