# FLARE-Federated-Learning-with-Accumulated-Regularized-Embeddings

This repository is our simulation for __

## Table of Contents

- [Abstract](#Abstract)
- [Directory Structure](#directory-structure)
- [Setup and Dependencies](#setup-and-dependencies)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Abstract

Federated Learning (FL) is an emerging paradigm that allows for decentralized machine learning (ML), where multiple models are collaboratively trained in a privacy-preserving manner. It has attracted much interest due to the significant advantages it brings to training deep neural network (DNN) models, particularly in terms of prioritizing privacy and enhancing the efficiency of communication resources when local data is stored at the edge devices. However, since communications and computation resources are limited, training DNN models in FL systems face challenges such as elevated computational and communication costs in complex tasks.

Sparse training schemes gain increasing attention in order to scale down the dimensionality of each client (i.e., node) transmission. Specifically, sparsification with error correction methods is a promising technique, where only important updates are sent to the parameter server (PS) and the rest are accumulated locally. While error correction methods have shown to achieve a significant sparsification level of the client-to-PS message without harming convergence, pushing sparsity further remains unresolved due to the staleness effect. In this paper, we propose a novel algorithm, dubbed Federated Learning with Accumulated Regularized Embeddings (FLARE), to overcome this challenge. FLARE presents a novel sparse training approach via accumulated pulling of the updated models with regularization on the embeddings in the FL process, providing a powerful solution to the staleness effect, and pushing sparsity to an exceptional level. The performance of FLARE is validated through extensive experiments on diverse and complex models, achieving a remarkable sparsity level (10 times and more beyond the current state-of-the-art) along with significantly improved accuracy. Additionally, an open-source software package has been developed for the benefit of researchers and developers in related fields.

## Directory Structure

Here's an overview of the project's directory structure:

- `data_handler/`: Raw and processed data files.
- `main/`: main script for experiments grid runs.
- `src/`: Python source code for federtaed training with FLARE algorithm and evaluation of Error Correction.
- `utils/`: utilities functions and congifs.
- `requirements.txt`: List of required Python packages.
- `README.md`: Documentation providing an overview of the project.

## Special Notes


## Modules

## Setup and Dependencies

To set up the project environment and follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/RanGreidi/FLARE.git

2. cd repo and edit main.py with the desired FLARE paramerters (for example, set Sparsity as 0.001%) .

3. run main.py.

4. main scripts iterates on experiments, each experiments runs on it's paramerters.

5. results are created at results directory for each experiment.



## Reconstruct Paper Results

To Reconstruct FLAER paper results, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/RanGreidi/FLARE.git

2. cd to repo and cd into paper_experiments _reconstruction/CNN or paper_experiments _reconstruction/VGG

3. run main.py

## Results
![Project Image](results/FC_0.001R_1E_0.5TAU_10CLIENTS_1001ROUNDS_1.05Decay_50u_OSR_1RegSteps.png)
