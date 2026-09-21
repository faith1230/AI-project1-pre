# Dynamic DQN on MountainCar-v0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-brightgreen.svg)](https://gymnasium.farama.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-Supported-orange.svg)](https://wandb.ai/)

本项目是一个基于 PyTorch 和 Gymnasium 实现的深度强化学习研究平台，专注于研究经典控制任务 **MountainCar-v0** 中深度 Q 网络（DQN）的**模型更新时机与更新频率策略**。

项目中对比了三种主流与前沿的训练更新策略：
1. **标准单步更新 DQN (Standard DQN)**：满足数据容量后每个交互步执行一次梯度更新。
2. **固定频率更新 DQN (Fixed-Frequency DQN)**：每隔固定步数（如每 1, 2, 4, 6, 8, 12, 16, 32 步）执行一次梯度更新。
3. **动态条件触发更新 DQN (Dynamic Condition DQN)**：依据价值函数与奖励的动态演化条件自适应决定是否触发梯度更新。

同时，平台支持**训练中周期性纯贪婪评测**、**全贪婪训练对照模式**、**Weights & Biases (WandB) 云端实验跟踪**以及**基于环境交互步的多维度可视化分析**。

---

## 目录

- [核心原理与动态更新条件](#核心原理与动态更新条件)
- [环境与奖励设计](#环境与奖励设计)
- [项目代码架构](#项目代码架构)
- [环境安装与依赖](#环境安装与依赖)
- [单元冒烟测试 (Smoke Tests)](#单元冒烟测试-smoke-tests)
- [完整实验与训练指南](#完整实验与训练指南)
  - [1. 算法训练 (Training)](#1-算法训练-training)
  - [2. 训练中周期性贪婪评估 (In-Training Evaluation)](#2-训练中周期性贪婪评估-in-training-evaluation)
  - [3. Weights & Biases (WandB) 实验追踪](#3-weights--biases-wandb-实验追踪)
  - [4. 独立模型评估 (Checkpoint Evaluation)](#4-独立模型评估-checkpoint-evaluation)
  - [5. 多种子聚合与 95% 置信区间对比 (Comparison)](#5-多种子聚合与-95-置信区间对比-comparison)
  - [6. 批量频率搜索实验 (Frequency Search)](#6-批量频率搜索实验-frequency-search)
  - [7. 实验可视化与图表绘制 (Visualization)](#7-实验可视化与图表绘制-visualization)
- [配置参数说明 (Hyperparameters)](#配置参数说明-hyperparameters)
- [终端实时监控仪表盘](#终端实时监控仪表盘)

---

## 核心原理与动态更新条件

在标准的强化学习训练过程中，网络通常在每个 step 都执行梯度下降，这在很多场景下存在大量的冗余梯度计算，并容易由于非关键转变造成过度拟合或目标漂移。

本项目提出了**动态条件更新机制**：
对于当前转移状态 $(S_t, A_t, R_t, S_{t+1})$，设状态价值函数为：
$$V(S) = \max_{a} Q(S, a)$$

动态更新判断准则定义为：
$$\text{sign}(V(S_{t+1})) \cdot \left[ V(S_t) - (R_t + V(S_{t+1})) \right] \ge 0$$

- 当满足上述不等式时，才从经验回放池（Replay Buffer）中采样并触发一步反向传播；
- 当不满足条件时跳过网络更新，保留现有特征直至遇到更具信息量或更符合收敛动态的经验。

---

## 环境与奖励设计

- **控制环境**: `MountainCar-v0`
  - 连续 2 维观测状态：小车水平位置 $[-1.2, 0.6]$、速度 $[-0.07, 0.07]$。
  - 离散动作空间：`0` 向左推力、`1` 无推力、`2` 向右推力。
  - 终止条件：小车到达山顶（位置 $\ge 0.5$）或达到环境最大截断步数（200步）。
- **奖励封装 (`GoalRewardWrapper`)**:
  - **稀疏奖励 (Sparse Reward，默认开启)**: 小车正常行进状态每步奖励为 `0.0`，成功登顶时获得 `+1.0` 奖励。
  - **传统奖励 (Standard Reward)**: 可通过 `--no-sparse-reward` 切换为 Gymnasium 原始每步 `-1.0` 的时间惩罚奖励。

---

## 项目代码架构

```text
AI-project1-pre/
├── configs/
│   ├── __init__.py
│   └── base_config.py             # 实验基础超参数配置 (BaseConfig Dataclass，支持评测与 WandB 配置)
├── experiments/
│   ├── run_frequency_search.py    # 批量网格搜索实验脚本 (多随机种子、多更新间隔)
│   └── plot_frequency_comparison.py # 更新频率敏感性对比绘图脚本 (带 95% CI 误差条)
├── src/
│   ├── __init__.py
│   ├── checkpoint.py              # 模型权重与元数据保存/加载
│   ├── compare_evaluations.py     # 多种子评估统计与 Bootstrap 95% CI 计算
│   ├── dqn_agent.py               # DQN Agent 实现 (Q 网络、Epsilon-Greedy、Huber Loss)
│   ├── environment.py             # 环境初始化与 GoalRewardWrapper 稀疏奖励封装
│   ├── evaluate.py                # 评估模块 (支持 evaluate_checkpoint 与在线 evaluate_agent)
│   ├── plot_results.py            # 训练曲线 (按 Steps 对齐) 与评估指标面板生成
│   ├── replay_buffer.py           # 均匀经验回放池 (Replay Buffer)
│   ├── smoke_test_agent.py        # Agent 网络与动作选择冒烟测试
│   ├── smoke_test_dynamic_condition.py # 动态更新条件数学测试
│   ├── smoke_test_replay.py       # 经验回放采样冒烟测试
│   ├── smoke_test_reward.py       # 稀疏奖励逻辑冒烟测试
│   ├── smoke_test_update.py       # 梯度反向传播与 Target 网络同步测试
│   ├── train.py                   # 标准 DQN 训练入口 (集成 periodic eval & wandb)
│   ├── train_dynamic.py           # 动态条件更新 DQN 训练入口 (集成 periodic eval & wandb)
│   ├── train_fixed_frequency.py   # 固定频率更新 DQN 训练入口 (集成 periodic eval & wandb)
│   ├── train_monitor.py           # 基于 Rich 的实时交互式终端训练进度监视器
│   └── utils.py                   # 全局随机种子控制等辅助工具
├── results/                       # 训练与评估产物输出目录
│   ├── <method>/seed_<seed>/
│   │   ├── checkpoint.pt          # 模型权重与状态
│   │   ├── episodes.csv           # 训练各回合数据 (回报、长度、探索率等)
│   │   ├── eval_during_train.csv  # 训练过程中周期性纯贪婪评估记录
│   │   └── summary.csv            # 训练最终参数汇总
│   ├── comparison.csv             # 跨算法多随机种子汇总对比
│   ├── frequency_search_comparison.csv # 频率搜索汇总对比
│   └── plots/                     # 自动生成的分析图表 (.png)
│       ├── comparison_summary_dashboard.png
│       ├── seed_level_distributions.png
│       ├── training_learning_curves_steps.png
│       └── train_vs_greedy_eval_curves_steps.png
└── requirements.txt               # 运行环境依赖列表
```

---

## 环境安装与依赖

本项目依赖 Python 3.10+ 环境。所有依赖项已写入 `requirements.txt`：

```bash
# 1. 克隆或进入项目根目录
cd AI-project1-pre

# 2. 创建并激活虚拟环境
# Linux / macOS:
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD):
python -m venv .venv
.venv\Scripts\activate.bat

# 3. 安装所有必需依赖
pip install -r requirements.txt
```

核心依赖项包括：
- `gymnasium[classic-control]>=1.0.0` & `pygame>=2.5.0`: 经典控制仿真环境
- `torch>=2.0.0`: 神经网络构建与梯度计算
- `numpy>=1.26.0`: 科学计算与 Bootstrap 统计
- `matplotlib>=3.8.0`, `pandas>=2.0.0`, `seaborn>=0.13.0`: 统计图表与曲线生成
- `rich>=13.0.0`: 终端高品质动态进度条与实时指标表格
- `wandb>=0.16.0`: Weights & Biases 云端实验跟踪（可选）

---

## 单元冒烟测试 (Smoke Tests)

在开始大规模实验前，可运行一系列轻量级单元冒烟测试以验证环境和算法组件：

```bash
# 测试 1: Q 网络架构与 epsilon-greedy 策略输出
python -m src.smoke_test_agent

# 测试 2: 动态更新条件逻辑数学准则
python -m src.smoke_test_dynamic_condition

# 测试 3: 经验回放池存入与采样
python -m src.smoke_test_replay

# 测试 4: 稀疏奖励与环境目标判断
python -m src.smoke_test_reward

# 测试 5: 单步梯度更新与 Target 权重同步
python -m src.smoke_test_update
```

---

## 完整实验与训练指南

### 1. 算法训练 (Training)

每个训练脚本均支持周期性在线贪婪评估、全程纯贪婪模式以及 WandB 同步。

#### (A) 训练动态条件 DQN (Dynamic Condition DQN)
```bash
python -m src.train_dynamic \
  --total-env-steps 100000 \
  --seed 42 \
  --name dynamic_condition
```

#### (B) 训练固定频率更新 DQN (Fixed-Frequency DQN)
通过 `--interval` 指定每次更新之间的交互步长（如 4 步）：
```bash
python -m src.train_fixed_frequency \
  --interval 4 \
  --total-env-steps 100000 \
  --seed 42 \
  --name fixed_freq_4
```

#### (C) 训练标准单步更新 DQN (Standard DQN Baseline)
```bash
python -m src.train \
  --total-env-steps 100000 \
  --seed 42 \
  --name standard_dqn_baseline
```

> **训练高级通用参数**：
> - `--eval-interval <int>`: 训练中每隔多少交互步运行一次贪婪评测（默认 `10000`；设为 `0` 可关闭）。
> - `--eval-episodes <int>`: 周期性评测的测试轮数（默认 `10` 回合）。
> - `--always-greedy` / `--no-always-greedy`: 是否训练全程使用纯贪婪策略 $\epsilon=0.0$（用于对比无探索时的策略演进）。
> - `--use-wandb` / `--no-use-wandb`: 是否启用 WandB 云端实时记录。
> - `--wandb-project <str>`: WandB 项目名称（默认 `MountainCar-DQN`）。
> - `--sparse-reward` / `--no-sparse-reward`: 控制是否采用稀疏目标奖励（默认开启）。
> - `--output-dir <path>`: 自定义结果保存路径（默认 `results/<name>/seed_<seed>`）。

训练产物包括：
- `checkpoint.pt`: 最终模型权重（在线网络、目标网络及训练参数）。
- `episodes.csv`: 每个 episode 的累计回报、步数、成功与否、即时 epsilon 及 loss。
- `eval_during_train.csv`: 周期性贪婪评测数据（包含各阶段的 `mean_return`, `success_rate`, `mean_episode_length`）。
- `summary.csv`: 训练最终参数汇总与完成度指标。

---

### 2. 训练中周期性贪婪评估 (In-Training Evaluation)

在强化学习中，训练过程中的 Episode Return 掺杂了 $\epsilon$-greedy 探索噪声，不能完全反映策略真实水平。
代码在训练主循环中内置了独立的评估环境 `eval_env`：
- 每隔 `--eval-interval` 步（默认 10,000 步），固定使用纯贪婪策略（$\epsilon=0.0$）执行 `--eval-episodes` 轮（默认 10 轮）。
- 评估结果记录在 `eval_during_train.csv` 中，并可由绘图脚本自动生成 **训练探索回报 vs. 贪婪评估回报** 对比曲线。

---

### 3. Weights & Biases (WandB) 实验追踪

本项目原生支持 WandB 云端仪表盘监控：
```bash
# 1. 初始化并登录 WandB (首次使用需要)
wandb login

# 2. 启动带 WandB 跟踪的训练任务
python -m src.train_dynamic \
  --total-env-steps 100000 \
  --seed 42 \
  --name dynamic_condition \
  --use-wandb \
  --wandb-project MountainCar-DQN
```

WandB 会统一以 `env_step` 作为横坐标记录：
- `train/episode_return`: 训练回合探索回报
- `train/episode_length`: 训练回合步长
- `train/epsilon`: 探索率衰减轨迹
- `train/loss`: Huber Loss 变化
- `eval/mean_return`: 贪婪评测平均回报
- `eval/success_rate`: 贪婪评测成功率
- `eval/mean_episode_length`: 贪婪评测平均耗步

---

### 4. 独立模型评估 (Checkpoint Evaluation)

使用训练保存的模型检查点在测试集上执行更大规模（如 100 回合）的确定性评估：

```bash
python -m src.evaluate \
  --checkpoint results/dynamic_condition/seed_42/checkpoint.pt \
  --episodes 100 \
  --evaluation-seed 10000
```

评估输出：
- `evaluation_episodes.csv`: 100 轮评估中每一轮的回报与长度。
- `evaluation_summary.csv`: 平均回报 (Mean Return)、标准差、登顶成功率 (Success Rate)、平均耗步等统计量。

---

### 5. 多种子聚合与 95% 置信区间对比 (Comparison)

收集各算法在不同随机种子（例如 seeds: 9, 99, 999, 9999, 99999）下的评估结果，使用 **Bootstrap 重采样（10,000 次）** 计算 95% 置信区间（95% CI）：

```bash
python -m src.compare_evaluations \
  --result-dirs \
    results/standard_dqn \
    results/fixed_freq_4 \
    results/fixed_freq_16 \
    results/dynamic_condition \
  --output results/comparison.csv
```

输出的 `results/comparison.csv` 包含各类方法的均值、标准差及 95% CI 上下界。

---

### 6. 批量频率搜索实验 (Frequency Search)

使用自动化脚本一键运行更新频率搜索网格实验（遍历指定多组种子与不同更新间隔，自动训练、评估并生成汇总对比）：

```bash
python -m experiments.run_frequency_search \
  --seeds 9 99 999 9999 99999 \
  --intervals 1 2 4 6 8 12 16 32 \
  --total-env-steps 100000 \
  --eval-episodes 100 \
  --results-dir results \
  --output results/frequency_search_comparison.csv
```

---

### 7. 实验可视化与图表绘制 (Visualization)

#### (A) 绘制全套指标对比面板与交互步对齐学习曲线
```bash
python -m src.plot_results \
  --comparison results/comparison.csv \
  --results-dir results \
  --output-dir results/plots
```
将在 `results/plots/` 下生成以下图表：
1. `training_learning_curves_steps.png`: **以环境交互步 (`env_step`) 为横轴**的训练回报平滑曲线（带 95% CI 误差阴影），消除不同算法因回合长短不一导致的横坐标偏移。
2. `train_vs_greedy_eval_curves_steps.png`: **训练回报 vs. 贪婪评估回报同图对比**（虚线为带探索训练回报，实线带圆点为纯贪婪评测指标），直观检验策略泛化与过拟合情况。
3. `comparison_summary_dashboard.png`: Mean Return、Success Rate、Episode Length 三联柱状图（带 95% CI 误差棒）。
4. `seed_level_distributions.png`: 各算法在多种子下的箱线图（Boxplot）与散点分布图（Stripplot）。

#### (B) 绘制固定频率 vs. 动态条件更新对比曲线
```bash
python -m experiments.plot_frequency_comparison \
  --input results/frequency_search_comparison.csv \
  --output results/frequency_comparison_plot.png
```
生成横坐标为更新步长间隔、纵坐标为测试得分/成功率的敏感度曲线，直观展示动态更新条件相比传统固定步长更新的优势区间。

---

## 配置参数说明 (Hyperparameters)

所有基础超参数定义在 `configs/base_config.py` 中：

| 参数名 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `env_id` | `"MountainCar-v0"` | Gymnasium 仿真环境名称 |
| `seed` | `42` | 训练全局随机种子 |
| `experiment_name` | `"standard_dqn_baseline"` | 实验名称与存储子目录名 |
| `total_env_steps` | `100_000` | 智能体与环境交互的总步数 |
| `replay_capacity` | `50_000` | 经验回放池最大容量 |
| `batch_size` | `64` | 每次梯度更新采样的 Transition 批次大小 |
| `learning_starts` | `1_000` | 触发梯度学习前的预填充步数 |
| `hidden_dim` | `64` | MLP 隐藏层节点数（2 层全连接） |
| `gamma` | `0.99` | 强化学习未来回报折扣因子 |
| `learning_rate` | `1e-3` | Adam 优化器学习率 |
| `gradient_clip_norm`| `10.0` | 梯度裁剪最大模长，防止梯度爆炸 |
| `target_sync_interval`| `100` | Target Network 权重硬拷贝同步间隔步数 |
| `epsilon_start` | `1.0` | $\epsilon$-greedy 初始探索率 |
| `epsilon_end` | `0.005` | $\epsilon$-greedy 衰减下限 |
| `epsilon_decay_steps`| `50_000` | 线性衰减至 `epsilon_end` 所需步数 |
| `sparse_reward` | `True` | 是否使用稀疏目标奖励封装 |
| **`always_greedy_training`** | `False` | 训练时是否全程使用纯贪婪策略 ($\epsilon=0.0$) |
| **`eval_interval`** | `10_000` | 训练中每隔多少步执行一次纯贪婪策略评估 |
| **`eval_episodes`** | `10` | 每次周期性评估的回合数 |
| **`eval_seed`** | `10_000` | 周期性评估基准随机种子 |
| **`use_wandb`** | `False` | 是否启用 Weights & Biases 实验跟踪 |
| **`wandb_project`** | `"MountainCar-DQN"` | WandB 云端项目名称 |

---

## 终端实时监控仪表盘

项目中集成了基于 `rich` 库的实时交互式控制台监视器（`src/train_monitor.py`）。
在训练过程中，无需切换第三方平台，即可在终端中实时查看：
- **总体交互步数进度条与预计剩余时间 (ETA)**
- **当前 Episode 编号、回合累计回报与长度**
- **即时 Epsilon 探索率与采样速率 (Steps/sec)**
- **经验池占用量 (Buffer Size / Capacity)**
- **平均预测 Q 值与 TD Target**
- **当前最新 Huber Loss**
- **贪心动作数 vs. 探索动作数**
- **动态条件触发状态 (Condition Triggers & Status)**

