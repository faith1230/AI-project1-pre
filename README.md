# Dynamic DQN project

Phase 1: verify the MountainCar environment and establish a reproducible experiment scaffold.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python -m src.train
```

The first run only executes a random-policy smoke test. DQN, replay memory, and the dynamic update condition will be added in later phases.
