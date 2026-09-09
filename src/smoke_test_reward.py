import numpy as np

from src.environment import make_env
from src.train_dynamic import dynamic_update_condition


def main() -> None:
    # 1. Test sparse reward (default)
    env_sparse = make_env("MountainCar-v0", seed=42, sparse_reward=True)
    state, _ = env_sparse.reset(seed=42)

    # Regular steps should yield reward 0.0
    for _ in range(5):
        action = 1  # no push
        state, reward, terminated, truncated, _ = env_sparse.step(action)
        assert reward == 0.0, f"Expected 0.0 for non-goal step, got {reward}"
        assert not terminated, "Did not expect termination at the valley bottom"

    # Manually place car at goal position (goal_position = 0.5)
    env_sparse.unwrapped.state = np.array([0.499, 0.05], dtype=np.float32)
    next_state, goal_reward, terminated, truncated, _ = env_sparse.step(2)  # push right
    assert terminated, "Expected episode to terminate upon reaching goal"
    assert goal_reward == 1.0, f"Expected reward 1.0 at goal, got {goal_reward}"
    env_sparse.close()

    # 2. Test standard reward when sparse_reward=False
    env_standard = make_env("MountainCar-v0", seed=42, sparse_reward=False)
    env_standard.reset(seed=42)
    _, std_reward, _, _, _ = env_standard.step(1)
    assert std_reward == -1.0, f"Expected -1.0 for standard reward step, got {std_reward}"
    env_standard.close()

    # 3. Verify dynamic condition compatibility with sparse reward
    # For positive value regime (since rewards are in [0, 1]):
    # sign(value) * (last_value - (reward + value)) >= 0
    # e.g., value = 0.5, last_value = 0.6, reward = 0.0 -> +1 * (0.6 - 0.5) = +0.1 >= 0 -> True
    assert dynamic_update_condition(last_value=0.6, reward=0.0, value=0.5) is True
    # e.g., value = 0.7, last_value = 0.5, reward = 0.0 -> +1 * (0.5 - 0.7) = -0.2 < 0 -> False
    assert dynamic_update_condition(last_value=0.5, reward=0.0, value=0.7) is False
    # At goal step: reward = 1.0, next_state terminal value = 0.0 -> sign(0)=0 -> 0 >= 0 -> True
    assert dynamic_update_condition(last_value=0.9, reward=1.0, value=0.0) is True

    print("Sparse goal-reward smoke tests passed successfully!")


if __name__ == "__main__":
    main()
