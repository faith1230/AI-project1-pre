import gymnasium as gym


class GoalRewardWrapper(gym.Wrapper):
    """
    Reward wrapper modifying the environment reward:
    - Normal transition: 0.0
    - Reaching the goal: +1.0
    """

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        goal_reached = bool(
            terminated
            or (
                hasattr(self.env.unwrapped, "goal_position")
                and state[0] >= self.env.unwrapped.goal_position
            )
        )
        custom_reward = 1.0 if goal_reached else 0.0
        return state, custom_reward, terminated, truncated, info


def make_env(env_id: str, seed: int, sparse_reward: bool = True):
    env = gym.make(env_id)
    if sparse_reward:
        env = GoalRewardWrapper(env)
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
