# Neurosmash
A plausible deep learning reinforcement agent for the Neurosmash environment.

## Deep Q-Learning Agent
Two reinforcement learning agents have been implemented. A conventional deep Q-learning agent and a biologically more plausible agent using [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). Both agents make use of the following techniques:
* Convolutional Layers
* Backpropagation
* [Frame Stacking](https://arxiv.org/abs/1312.5602)
* [LP Pooling](https://arxiv.org/abs/1204.3968)
* [Huber Loss](https://doi.org/10.1007/s00521-020-04741-w)
* [Frozen Target Network](https://arxiv.org/abs/1312.5602)

Aside from this, the vanilla implementations used
* [Memory Replay](https://doi.org/10.1109/ALLERTON.2018.8636075)

The biologically plausible agent implements 
* [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)

## Requirements
To run the program, make sure you have installed the dependencies listed in environment.yml. 
We recommend creating a conda environment for every project. You can do this with the following command:
`conda env create --file environment.yml`

## Usage
To run the networks take a look at the python notebooks in the src folder.
