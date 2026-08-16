# The Self-Pruning Neural Network

A PyTorch implementation of a self-pruning feed-forward neural network for CIFAR-10 image classification, developed as part of the Tredence Analytics AI Engineering Internship case study.

## Project Overview

The goal of this project is to build a neural network that learns which of its own connections are unnecessary during training.

Each weight in the network has a corresponding learnable gate score. The gate is obtained using a sigmoid function:

G = sigmoid(gate_scores)

The effective weight is:

W_effective = W \* G

A sparsity regularization term encourages unnecessary gates to become small.

The total training loss is:

Total Loss = Classification Loss + lambda \* Sparsity Loss

where:

Sparsity Loss = sum of all gate values

Connections whose gate value falls below 1e-2 are considered pruned during final evaluation.

## Architecture

The CIFAR-10 images have dimensions 3 x 32 x 32, giving 3072 input features after flattening.

The network architecture is:

```text
CIFAR-10 Image
      |
    Flatten
      |
    3072
      |
      v
PrunableLinear
  3072 -> 256
      |
     ReLU
      |
      v
PrunableLinear
   256 -> 128
      |
     ReLU
      |
      v
PrunableLinear
    128 -> 10
      |
      v
 Class Scores
```

All fully connected layers use the custom `PrunableLinear` implementation.

## Key Features

- Custom `PrunableLinear` layer implemented from scratch
- Learnable `gate_scores` with the same shape as the weight tensor
- Sigmoid-based differentiable gates
- L1 sparsity regularization
- Adam optimization
- CIFAR-10 training and evaluation
- Three different lambda experiments
- Hard pruning using a 1e-2 threshold
- Gradient-flow verification for weights and gate scores
- Gate distribution visualization
- Accuracy-vs-sparsity visualization

## Experimental Setup

| Parameter         |                    Value |
| ----------------- | -----------------------: |
| Dataset           |                 CIFAR-10 |
| Architecture      | 3072 -> 256 -> 128 -> 10 |
| Optimizer         |                     Adam |
| Learning Rate     |                    0.001 |
| Batch Size        |                      128 |
| Epochs            |                       30 |
| Pruning Threshold |                     0.01 |
| Lambda Values     |         1e-5, 5e-5, 1e-4 |

## Results

The experiments produced the following results:

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| -----: | ----------------: | -----------------: |
|   1e-5 |             53.65 |               3.37 |
|   5e-5 |             54.94 |              28.34 |
|   1e-4 |         **55.17** |          **46.07** |

The best result was obtained with lambda = 1e-4:

- Test Accuracy: **55.17%**
- Sparsity: **46.07%**

The results demonstrate that increasing lambda substantially increases sparsity while maintaining comparable classification accuracy over the tested range.

## Results Visualization

### Gate Distribution

The final gate distribution for the best model is available at:

`results/gate_distribution_best.png`

### Accuracy vs Sparsity

The accuracy-sparsity trade-off is available at:

`results/accuracy_vs_sparsity.png`

## Project Structure

```text
self-pruning-neural-network/
|
|-- train.py
|-- requirements.txt
|-- README.md
|-- report.md
|-- .gitignore
|
`-- results/
    |-- results.csv
    |-- gate_distribution_best.png
    `-- accuracy_vs_sparsity.png
```

The CIFAR-10 dataset, virtual environment, and trained model checkpoint are not required in the repository.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/YOUR_USERNAME/self-pruning-neural-network.git
cd self-pruning-neural-network
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run:

```bash
python train.py
```

The CIFAR-10 dataset will be downloaded automatically.

The script trains separate models for the configured lambda values and generates:

```text
results/results.csv
results/gate_distribution_best.png
results/accuracy_vs_sparsity.png
results/best_model.pth
```

The trained model checkpoint is intentionally kept out of GitHub because it is a generated binary artifact and is not required to reproduce the code or analysis.

## Report

The detailed case-study report is available in:

`report.md`

It contains the methodology, sparsity formulation, experimental results, lambda analysis, gate statistics, and conclusions.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- tqdm

## Case Study Objective

This implementation demonstrates how learnable gates and sparsity regularization can be combined to allow a neural network to identify and suppress unnecessary connections during training.

The project focuses on correctness, reproducibility, experimental analysis, and the accuracy-sparsity trade-off.
