import torch
import numpy as np
import matplotlib.pyplot as plt
import torch, numpy as np, random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def structured_lr_finder(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    lr_start=1e-6,
    lr_end=1.0,
    num_iter=200,
    beta=0.98,
    seed=42
):
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.train()
    lr_mult = (lr_end / lr_start) ** (1 / num_iter)
    lr = lr_start
    optimizer.param_groups[0]['lr'] = lr

    avg_loss, best_loss = 0., float('inf')
    losses, log_lrs = [], []

    iter_count = 0
    data_iter = iter(dataloader)

    for _ in range(num_iter):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, labels = next(data_iter)

        iter_count += 1
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Smooth loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed = avg_loss / (1 - beta ** iter_count)

        # Record even if exploding
        losses.append(smoothed)
        log_lrs.append(np.log10(lr))

        # Track best loss (for normalisation)
        if smoothed < best_loss:
            best_loss = smoothed

        # Backprop
        loss.backward()
        optimizer.step()

        # Increase LR
        lr *= lr_mult
        optimizer.param_groups[0]['lr'] = lr

        # Optional: stop if completely divergent
        if np.isnan(smoothed) or smoothed > best_loss * 1000:
            print("Loss diverged — stopping early.")
            break

    return log_lrs, losses

from supervised_learning.models.STAF_V2 import TemporalAttentiveFusionNet
from supervised_learning.loader import get_loaders_from_files
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalAttentiveFusionNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

train_loader, _, _ = get_loaders_from_files(
    merged_csv_path="supervised_learning/training_data/merged_windowed_dataset.csv",
    feature_info_path="supervised_learning/training_data/feature_info.json",
    sequence_length=24,
    batch_size=4,
    num_workers=2
)

log_lrs, losses = structured_lr_finder(
    model, train_loader, optimizer, criterion, DEVICE,
    lr_start=1e-6, lr_end=1, num_iter=500
)

plt.figure(figsize=(8,6), dpi=150)
plt.plot(log_lrs, losses)
plt.xlabel("log10(Learning Rate)")
plt.ylabel("Loss (smoothed)")
plt.yscale("log")
plt.title("Structured LR Finder")
plt.grid(alpha=0.3)
plt.savefig("./supervised_learning/output/lr_finder_plot.png")
plt.show()
