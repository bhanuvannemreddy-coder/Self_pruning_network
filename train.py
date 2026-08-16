"""
The Self-Pruning Neural Network
Tredence Analytics - AI Engineering Internship Case Study

Requirements implemented:
1. Custom PrunableLinear layer
2. Learnable gate_scores with same shape as weights
3. Sigmoid gates in [0, 1]
4. Effective weights = weight * gates
5. L1 sparsity loss = sum of all gate values
6. Total loss = CrossEntropy + lambda * SparsityLoss
7. Adam optimizer updates weights and gate_scores
8. CIFAR-10 training and testing
9. Three lambda experiments
10. Sparsity measured using gate < 1e-2
11. Hard-pruned model evaluation
12. Gate distribution plot
13. Accuracy-vs-sparsity plot
14. Gradient-flow verification
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Training
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3

# ------------------------------------------------------------
# We are using the RAW SUM of all gates as required by the
# assignment:
#
#     SparsityLoss = sum(gates)
#
# Since there are many gates, lambda must be small.
#
# These values are intentionally stronger than the previous
# experiment so that we can investigate a meaningful
# sparsity-vs-accuracy trade-off.
# ------------------------------------------------------------

LAMBDA_VALUES = [
    1e-5,
    5e-5,
    1e-4
]

# Assignment specifies 1e-2 as an example threshold
PRUNING_THRESHOLD = 1e-2

DATA_DIR = "./data"
RESULTS_DIR = "./results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PART 1
# PRUNABLE LINEAR LAYER
# ============================================================

class PrunableLinear(nn.Module):
    """
    Custom linear layer with one learnable gate score
    corresponding to every weight.

    Standard layer:

        y = Wx + b

    Prunable layer:

        gates = sigmoid(gate_scores)

        pruned_weights = weight * gates

        y = pruned_weights x + b
    """

    def __init__(self, in_features, out_features):

        super().__init__()

        # ----------------------------------------------------
        # Standard weight parameter
        # Shape:
        # [out_features, in_features]
        # ----------------------------------------------------

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features
            )
        )

        # ----------------------------------------------------
        # Standard bias parameter
        # ----------------------------------------------------

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # ----------------------------------------------------
        # Learnable gate scores
        #
        # EXACTLY the same shape as weight.
        # ----------------------------------------------------

        self.gate_scores = nn.Parameter(
            torch.zeros(
                out_features,
                in_features
            )
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        nn.init.kaiming_uniform_(
            self.weight,
            a=np.sqrt(5)
        )

    def get_gates(self):
        """
        Convert gate scores to values between 0 and 1.
        """

        return torch.sigmoid(
            self.gate_scores
        )

    def forward(self, x):
        """
        Differentiable gated linear operation.
        """

        # Convert gate scores to gates
        gates = self.get_gates()

        # Element-wise gating of weights
        pruned_weights = (
            self.weight * gates
        )

        # Linear operation
        return F.linear(
            x,
            pruned_weights,
            self.bias
        )


# ============================================================
# NETWORK
# ============================================================

class SelfPruningNetwork(nn.Module):
    """
    Feed-forward CIFAR-10 classifier.

    Architecture:

        3072 -> 256 -> 128 -> 10
    """

    def __init__(self):

        super().__init__()

        self.fc1 = PrunableLinear(
            3072,
            256
        )

        self.fc2 = PrunableLinear(
            256,
            128
        )

        self.fc3 = PrunableLinear(
            128,
            10
        )

    def forward(self, x):

        # CIFAR-10:
        # [batch, 3, 32, 32]
        #
        # Flatten:
        # [batch, 3072]

        x = x.view(
            x.size(0),
            -1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x


# ============================================================
# SPARSITY LOSS
# ============================================================

def calculate_sparsity_loss(model):
    """
    Calculate the L1 norm of all gate values.

    Since gates are positive:

        |gate| = gate

    Therefore:

        SparsityLoss = sum(gate)
    """

    sparsity_loss = torch.tensor(
        0.0,
        device=DEVICE
    )

    for module in model.modules():

        if isinstance(
            module,
            PrunableLinear
        ):

            gates = module.get_gates()

            sparsity_loss = (
                sparsity_loss
                +
                gates.sum()
            )

    return sparsity_loss


# ============================================================
# DATASET
# ============================================================

def get_dataloaders():
    """Load CIFAR-10 training and testing datasets."""

    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (
                0.4914,
                0.4822,
                0.4465
            ),
            (
                0.2470,
                0.2435,
                0.2616
            )
        )
    ])

    # Training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    # Test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    lambda_value
):
    """
    Train for one epoch.

    Total Loss:

        Classification Loss
        +
        lambda * Sparsity Loss
    """

    model.train()

    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    sparsity_loss_sum = 0.0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Classification loss
        classification_loss = criterion(
            outputs,
            labels
        )

        # Sparsity loss
        sparsity_loss = calculate_sparsity_loss(
            model
        )

        # ----------------------------------------------------
        # EXACT LOSS REQUIRED BY TREDENCE
        # ----------------------------------------------------

        total_loss = (
            classification_loss
            +
            lambda_value * sparsity_loss
        )

        # Backpropagation
        total_loss.backward()

        # Update weights, biases and gate_scores
        optimizer.step()

        total_loss_sum += total_loss.item()
        classification_loss_sum += (
            classification_loss.item()
        )
        sparsity_loss_sum += (
            sparsity_loss.item()
        )

        progress.set_postfix(
            loss=f"{total_loss.item():.3f}",
            cls=f"{classification_loss.item():.3f}",
            sparse=f"{sparsity_loss.item():.0f}"
        )

    n = len(train_loader)

    return (
        total_loss_sum / n,
        classification_loss_sum / n,
        sparsity_loss_sum / n
    )


# ============================================================
# SOFT-GATED EVALUATION
# ============================================================

def evaluate_soft(
    model,
    test_loader
):
    """
    Evaluate using the learned sigmoid gates directly.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    return (
        100.0 * correct / total
    )


# ============================================================
# HARD-PRUNED EVALUATION
# ============================================================

def evaluate_pruned(
    model,
    test_loader,
    threshold=PRUNING_THRESHOLD
):
    """
    Evaluate the model after applying hard pruning.

    Rule:

        gate < threshold  -> remove connection
        gate >= threshold -> retain connection
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        # ----------------------------------------------------
        # Construct binary masks ONCE.
        #
        # They remain fixed during evaluation.
        # ----------------------------------------------------

        gates1 = model.fc1.get_gates()
        gates2 = model.fc2.get_gates()
        gates3 = model.fc3.get_gates()

        mask1 = (
            gates1 >= threshold
        ).float()

        mask2 = (
            gates2 >= threshold
        ).float()

        mask3 = (
            gates3 >= threshold
        ).float()

        # Effective hard-pruned weights
        weights1 = (
            model.fc1.weight
            * gates1
            * mask1
        )

        weights2 = (
            model.fc2.weight
            * gates2
            * mask2
        )

        weights3 = (
            model.fc3.weight
            * gates3
            * mask3
        )

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Flatten
            x = images.view(
                images.size(0),
                -1
            )

            # Layer 1
            x = F.linear(
                x,
                weights1,
                model.fc1.bias
            )

            x = F.relu(x)

            # Layer 2
            x = F.linear(
                x,
                weights2,
                model.fc2.bias
            )

            x = F.relu(x)

            # Layer 3
            outputs = F.linear(
                x,
                weights3,
                model.fc3.bias
            )

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    return (
        100.0 * correct / total
    )


# ============================================================
# SPARSITY MEASUREMENT
# ============================================================

def calculate_sparsity(
    model,
    threshold=PRUNING_THRESHOLD
):
    """
    Calculate:

        Sparsity =
        percentage of gates below threshold
    """

    total_gates = 0
    pruned_gates = 0

    all_gates = []

    with torch.no_grad():

        for module in model.modules():

            if isinstance(
                module,
                PrunableLinear
            ):

                gates = module.get_gates()

                all_gates.append(
                    gates.cpu().flatten()
                )

                total_gates += (
                    gates.numel()
                )

                pruned_gates += (
                    gates < threshold
                ).sum().item()

    sparsity = (
        100.0
        *
        pruned_gates
        /
        total_gates
    )

    all_gates = torch.cat(
        all_gates
    )

    return sparsity, all_gates


# ============================================================
# GRADIENT VERIFICATION
# ============================================================

def verify_gradient_flow(model):
    """
    Verify that both weights and gate_scores receive
    gradients.

    This directly addresses the custom-layer requirement.
    """

    print("\n" + "-" * 60)
    print("GRADIENT FLOW CHECK")
    print("-" * 60)

    for name in [
        "fc1",
        "fc2",
        "fc3"
    ]:

        layer = getattr(
            model,
            name
        )

        weight_ok = (
            layer.weight.grad is not None
        )

        gate_ok = (
            layer.gate_scores.grad is not None
        )

        print(
            f"{name}: "
            f"weight_grad={weight_ok}, "
            f"gate_score_grad={gate_ok}"
        )

        if not weight_ok:
            raise RuntimeError(
                f"Weight gradient missing: {name}"
            )

        if not gate_ok:
            raise RuntimeError(
                f"Gate gradient missing: {name}"
            )


# ============================================================
# GATE DISTRIBUTION
# ============================================================

def plot_gate_distribution(
    gates,
    lambda_value,
    output_path
):
    """
    Plot the distribution of final gate values.
    """

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        gates.numpy(),
        bins=100
    )

    plt.axvline(
        PRUNING_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=(
            f"Threshold = "
            f"{PRUNING_THRESHOLD}"
        )
    )

    plt.xlabel(
        "Gate Value"
    )

    plt.ylabel(
        "Number of Gates"
    )

    plt.title(
        "Final Gate Value Distribution\n"
        f"Lambda = {lambda_value}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# ACCURACY VS SPARSITY
# ============================================================

def plot_accuracy_vs_sparsity(
    results_df,
    output_path
):
    """
    Plot test accuracy against sparsity level.
    """

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results_df["Sparsity Level (%)"],
        results_df["Test Accuracy"],
        marker="o"
    )

    # Label each point with its lambda
    for _, row in results_df.iterrows():

        plt.annotate(
            f"λ={row['Lambda']:.0e}",
            (
                row["Sparsity Level (%)"],
                row["Test Accuracy"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel(
        "Sparsity Level (%)"
    )

    plt.ylabel(
        "Test Accuracy (%)"
    )

    plt.title(
        "Accuracy vs Sparsity"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THE SELF-PRUNING NEURAL NETWORK")
    print("Tredence Analytics Case Study")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch Size: {BATCH_SIZE}"
    )

    print(
        f"Learning Rate: {LEARNING_RATE}"
    )

    print(
        f"Lambda Values: {LAMBDA_VALUES}"
    )

    print(
        f"Pruning Threshold: "
        f"{PRUNING_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading CIFAR-10...")

    train_loader, test_loader = (
        get_dataloaders()
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_loader.dataset)}"
    )

    criterion = nn.CrossEntropyLoss()

    results = []

    # Best model = highest hard-pruned accuracy
    best_model = None
    best_lambda = None
    best_accuracy = -1
    best_sparsity = 0
    best_gates = None

    # ========================================================
    # THREE LAMBDA EXPERIMENTS
    # ========================================================

    for lambda_value in LAMBDA_VALUES:

        print("\n")
        print("=" * 70)
        print(
            f"EXPERIMENT: λ = {lambda_value:.0e}"
        )
        print("=" * 70)

        # Fresh model for fair comparison
        set_seed()

        model = (
            SelfPruningNetwork()
            .to(DEVICE)
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for epoch in range(EPOCHS):

            (
                total_loss,
                classification_loss,
                sparsity_loss
            ) = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                lambda_value
            )

            soft_accuracy = (
                evaluate_soft(
                    model,
                    test_loader
                )
            )

            current_sparsity, _ = (
                calculate_sparsity(
                    model
                )
            )

            print(
                f"Epoch "
                f"{epoch + 1:02d}/{EPOCHS} | "
                f"Total Loss: "
                f"{total_loss:.4f} | "
                f"CE: "
                f"{classification_loss:.4f} | "
                f"Sparsity Loss: "
                f"{sparsity_loss:.1f} | "
                f"Accuracy: "
                f"{soft_accuracy:.2f}% | "
                f"Sparsity: "
                f"{current_sparsity:.2f}%"
            )

        # ----------------------------------------------------
        # Verify gradient flow
        # ----------------------------------------------------

        verify_gradient_flow(model)

        # ----------------------------------------------------
        # Final soft accuracy
        # ----------------------------------------------------

        soft_accuracy = (
            evaluate_soft(
                model,
                test_loader
            )
        )

        # ----------------------------------------------------
        # Final sparsity
        # ----------------------------------------------------

        sparsity, gates = (
            calculate_sparsity(
                model
            )
        )

        # ----------------------------------------------------
        # Accuracy after hard pruning
        # ----------------------------------------------------

        pruned_accuracy = (
            evaluate_pruned(
                model,
                test_loader
            )
        )

        # ----------------------------------------------------
        # Print final results
        # ----------------------------------------------------

        print("\n" + "-" * 60)
        print(
            f"FINAL RESULT: "
            f"λ = {lambda_value:.0e}"
        )
        print("-" * 60)

        print(
            f"Soft Accuracy: "
            f"{soft_accuracy:.2f}%"
        )

        print(
            f"Hard-Pruned Accuracy: "
            f"{pruned_accuracy:.2f}%"
        )

        print(
            f"Sparsity: "
            f"{sparsity:.2f}%"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "Lambda": lambda_value,
            "Test Accuracy": pruned_accuracy,
            "Soft Accuracy": soft_accuracy,
            "Sparsity Level (%)": sparsity
        })

        # ----------------------------------------------------
        # Select best model
        # ----------------------------------------------------

        if pruned_accuracy > best_accuracy:

            best_accuracy = (
                pruned_accuracy
            )

            best_lambda = (
                lambda_value
            )

            best_sparsity = (
                sparsity
            )

            best_model = model

            best_gates = (
                gates.clone()
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_path = os.path.join(
        RESULTS_DIR,
        "best_model.pth"
    )

    torch.save(
        {
            "model_state_dict":
                best_model.state_dict(),

            "lambda":
                best_lambda,

            "test_accuracy":
                best_accuracy,

            "sparsity":
                best_sparsity,

            "threshold":
                PRUNING_THRESHOLD
        },
        model_path
    )

    # ========================================================
    # BEST MODEL GATE DISTRIBUTION
    # ========================================================

    gate_plot_path = os.path.join(
        RESULTS_DIR,
        "gate_distribution_best.png"
    )

    plot_gate_distribution(
        best_gates,
        best_lambda,
        gate_plot_path
    )

    # ========================================================
    # ACCURACY VS SPARSITY PLOT
    # ========================================================

    tradeoff_plot_path = os.path.join(
        RESULTS_DIR,
        "accuracy_vs_sparsity.png"
    )

    plot_accuracy_vs_sparsity(
        results_df,
        tradeoff_plot_path
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)

    print("\nRequired Tredence Results:")
    print()

    print(
        results_df[
            [
                "Lambda",
                "Test Accuracy",
                "Sparsity Level (%)"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nBest Model")
    print("-" * 40)

    print(
        f"Lambda: "
        f"{best_lambda:.0e}"
    )

    print(
        f"Test Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Sparsity: "
        f"{best_sparsity:.2f}%"
    )

    print("\nGenerated files:")

    print(
        f"- {results_path}"
    )

    print(
        f"- {model_path}"
    )

    print(
        f"- {gate_plot_path}"
    )

    print(
        f"- {tradeoff_plot_path}"
    )

    print("\nDone!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()