# zig-NEAT

[NeuroEvolution — evolving Artificial Neural Networks topology from scratch](https://becominghuman.ai/neuroevolution-evolving-artificial-neural-networks-topology-from-the-scratch-d1ebc5540d84)

### Project is a Work In Progress

This project is a work in progress; expect the API to change. For bugs/feature request, please file an issue.

## Overview

This repository provides implementation of [NeuroEvolution of Augmenting Topologies (NEAT)](http://www.cs.ucf.edu/~kstanley/neat.html) method written in Zig.

The NeuroEvolution (NE) is an artificial evolution of Neural Networks (NN) using genetic algorithms to find optimal NN parameters and network topology. NeuroEvolution of NN may assume a search for optimal weights of connections between NN nodes and search for the optimal topology of the resulting network graph. The NEAT method implemented in this work searches for optimal connection weights and the network graph topology for a given task (number of NN nodes per layer and their interconnections).

Specifically, this project aims to port the [goNEAT](https://github.com/yaricom/goNEAT) Golang library, which implements NEAT in Go and is what originally sparked my interest in the subject of NEAT.

## Minimum Requirements

| Requirement | Notes  |
| ----------- | ------ |
| Zig version | master |

I recommend using [zigup](https://github.com/marler8997/zigup) to manage the version of Zig running locally.

To install the relevant build of Zig:

```bash
zigup master
```

## Run Examples

**TODO: Add Context/Findings for each of the following examples**

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

To run the Cart2Pole (Markov) example, run the following command:

```bash
cd examples/cart2pole
# alternatively, use -Doptimize=ReleaseFast
zig build run -Doptimize=ReleaseSafe
```

To run the Maze (Novelty Search Based Optimization) example with medium difficulty map:

```bash
cd examples/maze
zig build run -Doptimize=ReleaseSafe -- --out out/medium_mazens --context data/maze.neat --genome data/mazestartgenes --maze data/medium_maze.txt --experiment MazeNS
```

To run the Maze (Novelty Search Based Optimization) example with hard difficulty map:

```bash
cd examples/maze
zig build run -Doptimize=ReleaseSafe -- --out out/hard_mazens --context data/maze.neat --genome data/mazestartgenes --maze data/hard_maze.txt --experiment MazeNS
```

To run the Maze (Objective Based Optimization) example with medium difficulty map:

```bash
cd examples/maze
zig build run -Doptimize=ReleaseSafe -- --out out/medium_mazeobj --context data/maze.neat --genome data/mazestartgenes --maze data/medium_maze.txt --experiment MazeOBJ
```

To run the Maze (Objective Based Optimization) example with hard difficulty map:

```bash
cd examples/maze
zig build run -Doptimize=ReleaseSafe -- --out out/hard_mazeobj --context data/maze.neat --genome data/mazestartgenes --maze data/hard_maze.txt --experiment MazeOBJ
```

## Roadmap

- [x] Implement basic Graph Theory functionality
- [x] Implement basic NEAT Algorithm
  - [x] Working Sequential Population Epoch Executor (single-threaded species reproduction)
  - [x] Working Parallel Population Epoch Executor (multi-threaded; thread per species reproduction)
  - [ ] Basic NEAT Examples
    - [x] XOR Connected
    - [x] Cartpole
    - [x] Cart2pole (Markov)
    - [ ] Cart2pole (Non-Markov)
    - [x] Maze (Novelty Search Based Optimization)
    - [x] Maze (Objective Based Fitness Optimization)
    - [ ] Retina (ES-HyperNEAT)
- [x] Implement Novelty Search Optimization
- [x] Implement HyperNEAT Algorithm
- [x] Implement ES-HyperNEAT Algorithm
- [ ] Export resulting Artificial Neural Network in format compatible w popular ML libraries? (i.e. allow saving the "best" model so that it can be loaded and used via an optimized ML framework)
