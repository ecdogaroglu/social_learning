# Social Learning via Multi-Agent Reinforcement Learning

This repository implements a deep reinforcement learning approach to social learning in networks, as part of my ongoing master's thesis. The implementation handles both single-agent and multi-agent cases with a binary state space and uses recurrent neural networks (RNNs) to process sequential observations.

## Theory Background

The model considers agents who must learn the true state of the world through:
- Private signals received in each period
- Observations of other agents' actions
- Unobservable rewards

Key theoretical aspects:
- Agents receive signals with accuracy q > 0.5
- The true state ω remains fixed throughout the learning process
- There exists a theoretical upper bound (r_bdd) on the learning rate

## Multi-Agent Implementations

The repository implements two distinct approaches to multi-agent learning:

### 1. Centralized Training with Decentralized Execution (CTDE)

The CTDE approach features:
- Decentralized actors that make independent decisions
- A centralized critic that has access to all agents' information
- Shared value function learning across agents
- Coordinated policy updates using global information

Key components:
- `DecentralizedActor`: Individual policy networks for each agent
- `CentralizedCritic`: Joint value function estimation
- `MultiAgentTrainer`: Coordinates training across all agents

### 2. Decentralized Training with Decentralized Execution (DTDE)

The DTDE approach implements:
- Fully independent agents with local training
- Individual actor-critic architectures
- Local value function estimation
- Independent policy updates

Key components:
- `DecentralizedAgent`: Complete agent with local actor, critic, and RNN
- Separate optimizers for policy, value, and RNN networks
- Independent learning without shared information

## Implementation Details

### Environment

- Binary state space (ω ∈ {0,1})
- Signal accuracy parameter q (default: 0.75)
- Rewards as a function of signal and action
- Support for multiple agents with shared observations

### Neural Architecture

The implementation uses several neural components:

1. **Actor Networks**
   - GRU-based RNN for processing observation history
   - Policy head that outputs action probabilities
   - Parameters optimized using policy gradient

2. **Critic Networks**
   - CTDE: Global critic with access to all agents' states
   - DTDE: Local critics for independent value estimation
   - Value heads for computing state-value estimates
   - Trained using temporal difference learning

3. **Multi-Agent RNN**
   - Processes signal and action histories
   - Handles observations from multiple agents
   - Maintains hidden states for sequential learning

### Key Features

- Custom metrics tracking for mistake rates and learning rates
- Visualization tools for comparing empirical performance to theoretical bounds
- Implementation of the paper's reward function for binary case
- Efficient memory management using RNNs instead of growing state spaces
- Support for both centralized and decentralized training approaches

## Requirements

```
numpy
torch
matplotlib
```

## Usage

Basic usage example for both multi-agent approaches:

```python
# Train using CTDE approach
trained_model_ctde, metrics_ctde = train_ctde(
    num_agents=4,
    num_steps=10000,
    signal_accuracy=0.75
)

# Train using DTDE approach
trained_model_dtde, metrics_dtde = train_dtde(
    num_agents=4,
    num_steps=10000,
    signal_accuracy=0.75
)

# Plot learning curves for both approaches
plot_multi_agent_metrics(metrics_ctde, signal_accuracy=0.75)
plot_multi_agent_metrics(metrics_dtde, signal_accuracy=0.75)
```

## Implementation Classes

### MultiAgentEnvironment

Implements the binary state environment with:
- Fixed true state
- Signal generation based on accuracy parameter
- Reward computation for multiple agents
- Support for simultaneous agent actions

### Neural Components

#### CTDE Implementation
- `DecentralizedActor`: Individual policy networks
- `CentralizedCritic`: Global value estimation
- `MultiAgentTrainer`: Training coordination

#### DTDE Implementation
- `DecentralizedAgent`: Complete independent agents
- `DecentralizedActor`: Local policy networks
- `DecentralizedCritic`: Local value estimation

### MultiAgentMetricsTracker

Tracks and computes various performance metrics for plotting, including:
- Signal realizations
- Observed rewards
- Policies 
- Learning rates
