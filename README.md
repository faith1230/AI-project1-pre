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

train：
python -m src.train_dynamic \
  --total-env-steps 5000 \
  --seed 0 \
  --name dynamic_condition_smoke

  evalution:
  python -m src.evaluate \
  --checkpoint results/standard_dqn_smoke/seed_0/checkpoint.pt \
  --episodes 100 \
  --evaluation-seed 10000

  compare:
  python -m src.compare_evaluations \
  --result-dirs \
    results/standard_dqn_smoke \
    results/fixed_frequency_4_smoke \
    results/fixed_frequency_16_smoke \
    results/dynamic_condition_smoke \
  --output results/smoke_comparison.csv
