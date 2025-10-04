# EMDSAC-ft: Bridging the Gap in Offline-to-Online Reinforcement Learning through Value Distribution Learning

This repository contains the official implementation of **EMDSAC-ft**, a novel algorithm for offline-to-online reinforcement learning that addresses key challenges in both offline pre-training and online fine-tuning phases.

## 🚀 Key Features

- **Uncertainty Decoupling**: Separates epistemic and aleatoric uncertainty for better offline RL performance
- **Distributional Value Learning**: Captures full return distributions instead of just expected values
- **Efficient Fine-tuning**: UDPE and TTRPI modules for stable online adaptation
- **State-of-the-art Performance**: 14.9% average improvement over baselines, 25.8% improvement in fine-tuning

## 🏗️ Repository Structure

```
EMDSAC-ft/
├── Independent/                    # Independent training implementation
│   ├── example_train/            # Main training scripts
│   │   ├── configs/              # Configuration files
│   │   │   ├── offline/          # Offline training configs
│   │   │   └── ft/              # Fine-tuning configs
│   │   ├── networks/             # Neural network architectures
│   │   ├── training/             # Training modules
│   │   └── utils/                # Utility functions
│   └── env_gym/                  # Environment implementations
├── Vectorized/                   # Vectorized implementation
│   ├── Algorithms/               # Algorithm implementations
│   ├── configs/                 # Configuration files
│   └── main.py                   # Main training script
└── requirements.txt              # Dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda environment manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/dksen/EMDSAC-ft.git
cd EMDSAC-ft
```

2. **Create and activate conda environment:**
```bash
conda create -n EMDSAC python=3.8
conda activate EMDSAC
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Key Dependencies

- `torch>=1.9.0` - PyTorch framework
- `gym>=0.21.0` - OpenAI Gym environments
- `d4rl>=1.1.0` - D4RL offline RL datasets
- `mujoco-py>=2.0.0` - MuJoCo physics engine
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.4.0` - Visualization
- `tensorboard>=2.7.0` - Training monitoring

## 🎯 Quick Start

### Offline Pre-training

**Independent Implementation:**
```bash
conda activate EMDSAC
cd Independent/example_train
python train_ORL.py --config configs/offline/walker2d-medium-replay-v2.yaml
```

**Vectorized Implementation:**
```bash
conda activate EMDSAC
cd Vectorized
python main.py --config configs/walker2d-medium-replay-v2.yaml
```

### Online Fine-tuning

```bash
conda activate EMDSAC
cd Independent/example_train
python train_O2O.py --config configs/ft/walker2d-medium-replay-v2.yaml
```

## 📊 Algorithm Overview

### EMDSAC (Offline Pre-training)

**Core Components:**
1. **Ensemble Value Distribution Networks**: Quantify epistemic uncertainty from OOD actions
2. **Distributional Value Learning**: Capture aleatoric uncertainty from environmental randomness
3. **Pessimistic Value Iteration**: Select minimal expected values across ensemble

**Key Innovation:**
- Decouples epistemic and aleatoric uncertainty
- Uses distributional RL to model full return distributions
- Reduces ensemble complexity while maintaining performance

### EMDSAC-ft (Online Fine-tuning)

**Core Components:**
1. **UDPE (Uneven Distribution of Pessimism Elimination)**: Reduces bias in value estimates
2. **TTRPI (True Trust Region Policy Improvement)**: Adaptive policy constraints based on Bellman error

**Key Innovation:**
- Dynamically adjusts policy update strength
- Prevents performance collapse during fine-tuning
- Maintains sample efficiency in online settings
