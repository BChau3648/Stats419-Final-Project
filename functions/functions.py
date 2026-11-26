import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_mnist_base():
    """Downloads MNIST and normalizes pixel values."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset

def build_custom_dataloader(
    base_dataset,
    num_images=20000,
    noise_rate=0.0,
    imbalance_ratio=1.0,
    batch_size=128,
    seed=42,
):
    """Subsamples MNIST to exact N, applies configurable majority:minority class

    imbalance across half the classes, and injects symmetric label noise.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(base_dataset.targets)
    selected_indices = []
    minority_classes = []

    if imbalance_ratio <= 1.0:
        per_class = num_images // 10
        diff = num_images - (per_class * 10)
        for cls in range(10):
            cls_idx = np.where(targets == cls)[0]
            count = per_class + (diff if cls == 0 else 0)
            chosen = rng.choice(cls_idx, size=count, replace=False)
            selected_indices.extend(chosen)
    else:
        # Randomly choose 5 classes to be the downsampled minority classes
        minority_classes = sorted(rng.choice(10, size=5, replace=False).tolist())
        minority_set = set(minority_classes)
        majority_classes = [c for c in range(10) if c not in minority_set]

        # Total N = 5 * (ratio * k) + 5 * k = 5 * (ratio + 1) * k
        k = num_images / (5.0 * (imbalance_ratio + 1.0))
        minority_count = max(1, int(round(k)))
        majority_count = int(round(imbalance_ratio * k))
        diff = num_images - (5 * majority_count + 5 * minority_count)

        for cls in range(10):
            cls_idx = np.where(targets == cls)[0]
            count = minority_count if cls in minority_set else majority_count
            # Apply rounding adjustment to the first majority class
            if cls == majority_classes[0]:
                count += diff
            chosen = rng.choice(cls_idx, size=count, replace=False)
            selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    x_data = torch.stack([base_dataset[i][0] for i in selected_indices])
    y_data = torch.tensor(targets[selected_indices], dtype=torch.long)

    if noise_rate > 0.0:
        num_noisy = int(len(y_data) * noise_rate)
        noisy_indices = rng.choice(len(y_data), size=num_noisy, replace=False)
        for idx in noisy_indices:
            orig_label = y_data[idx].item()
            possible_labels = [c for c in range(10) if c != orig_label]
            y_data[idx] = rng.choice(possible_labels)

    custom_dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    return loader, minority_classes

class DynamicMLP(nn.Module):

    def __init__(
        self,
        input_dim=784,
        num_classes=10,
        hidden_layers=2,
        nodes_per_layer=256,
        dropout_rate=0.2,
    ):
        super().__init__()
        layers = [nn.Flatten()]

        in_features = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, nodes_per_layer))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = nodes_per_layer

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def evaluate_config_detailed(train_base, test_loader, config, epochs=10, seed=42):
    set_seed(seed)

    # 1. Unpack both the DataLoader and the randomly selected minority class IDs
    train_loader, minority_classes = build_custom_dataloader(
        train_base,
        num_images=config["num_images"],
        noise_rate=config["noise_rate"],
        imbalance_ratio=config["imbalance_ratio"],
        batch_size=config["batch_size"],
        seed=seed,
    )

    model = DynamicMLP(
        hidden_layers=config["hidden_layers"],
        nodes_per_layer=config["nodes_per_layer"],
        dropout_rate=config["dropout_rate"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    conf_matrix = torch.zeros(10, 10, dtype=torch.long)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            total_loss += criterion(outputs, y_batch).item() * y_batch.size(0)
            preds = outputs.argmax(dim=1)
            for t, p in zip(y_batch.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    tp = conf_matrix.diag().float()
    per_class_recall = tp / conf_matrix.sum(dim=1).float().clamp(min=1)
    per_class_precision = tp / conf_matrix.sum(dim=0).float().clamp(min=1)
    per_class_f1 = (
        2
        * (per_class_precision * per_class_recall)
        / (per_class_precision + per_class_recall).clamp(min=1e-8)
    )

    balanced_acc = per_class_recall.mean().item() * 100.0
    macro_f1 = per_class_f1.mean().item() * 100.0
    avg_test_loss = total_loss / conf_matrix.sum().item()

    # 2 & 3. Compute recall specifically on the randomly chosen minority classes
    if len(minority_classes) > 0:
        minority_recall = per_class_recall[minority_classes].mean().item() * 100.0
    else:
        minority_recall = balanced_acc  # All 10 classes are equally balanced

    # 4. Return minority_classes so each run records which digits were downsampled
    return {
        "balanced_acc": balanced_acc,
        "minority_recall": minority_recall,
        "macro_f1": macro_f1,
        "test_loss": avg_test_loss,
        "minority_classes": minority_classes,
    }

def recommend_binary_levels(results, primary_metric="macro_f1", min_valid_score=65.0):
    """Selects the two candidate levels that maximize separation on the primary_metric

    while filtering out diverged/collapsed runs (< min_valid_score).
    Each entry in `results` is (val, metrics_dict).
    """
    valid = [(val, m) for val, m in results if m[primary_metric] >= min_valid_score]
    if len(valid) < 2:
        valid = results

    low_pair = min(valid, key=lambda x: x[1][primary_metric])
    high_pair = max(valid, key=lambda x: x[1][primary_metric])
    return low_pair, high_pair