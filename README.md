# zig-NEAT

[NeuroEvolution — evolving Artificial Neural Networks topology from scratch](https://becominghuman.ai/neuroevolution-evolving-artificial-neural-networks-topology-from-the-scratch-d1ebc5540d84)

### Project is a Work In Progress

This project is a work in progress and is not yet ready for use.

For the brave few that want to try it out in the meantime, please file issues for any bugs, feature request, or API changes you'd like to see made.

## Overview

This repository provides implementation of [NeuroEvolution of Augmenting Topologies (NEAT)](http://www.cs.ucf.edu/~kstanley/neat.html) method written in Zig.

The NeuroEvolution (NE) is an artificial evolution of Neural Networks (NN) using genetic algorithms to find optimal NN parameters and network topology. NeuroEvolution of NN may assume a search for optimal weights of connections between NN nodes and search for the optimal topology of the resulting network graph. The NEAT method implemented in this work searches for optimal connection weights and the network graph topology for a given task (number of NN nodes per layer and their interconnections).

Specifically, this project aims to port the [goNEAT](https://github.com/yaricom/goNEAT) Golang library, which implements NEAT in Go and is what originally sparked my interest in the subject of NEAT.

## Minimum Requirements

| Requirement | Notes   |
| ----------- | ------- |
| Zig version | Nightly |

I recommend using [zigup](https://github.com/marler8997/zigup) to manage the version of Zig running locally.

## Run Examples

To run the XOR example, run the following command:

```bash
cd examples/xor
# alternatively, use -Doptimize=ReleaseFast
zig build run -Doptimize=ReleaseSafe
```

To run the CartPole example, run the following command:

```bash
cd examples/cartpole
# alternatively, use -Doptimize=ReleaseFast
zig build run -Doptimize=ReleaseSafe
```

## Roadmap

- [x] Implement basic Graph Theory functionality
- [x] Implement basic NEAT Algorithm
- [x] Working Sequential Population Epoch Executor (single-threaded species reproduction)
- [ ] Working Parallel Population Epoch Executor (multi-threaded; thread per species reproduction)
- [ ] Implement Novelty Search Optimization
- [ ] Implement HyperNEAT Algorithm
