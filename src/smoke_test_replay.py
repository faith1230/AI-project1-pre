import numpy as np

from configs.base_config import BaseConfig
from src.replay_buffer import ReplayBuffer


def main() -> None:
    config = BaseConfig()
    buffer = ReplayBuffer(capacity=32, seed=config.seed)

    for step in range(40):
        state = np.array([step, step + 0.1], dtype=np.float32)
        next_state = state + 1.0
        buffer.push(
            state=state,
            action=step % 3,
            reward=-1.0,
            next_state=next_state,
            terminated=(step == 19),
            truncated=(step == 39),
        )

    batch = buffer.sample(batch_size=8)
    assert len(buffer) == 32
    assert len(batch) == 8
    assert all(item.state.shape == (2,) for item in batch)
    assert all(isinstance(item.action, int) for item in batch)
    assert any(item.truncated for item in buffer.memory)

    print("Replay buffer smoke test passed")
    print("Current buffer size:", len(buffer))
    print("Sample batch size:", len(batch))
    print("One sampled transition:", batch[0])


if __name__ == "__main__":
    main()
