import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from datetime import datetime

# === import local modules ===
from dataloader.loader import get_loaders_from_files
from models.temporal_attentive_fusion_net_light import TemporalAttentiveFusionNet

# === Configs ===
RUN_NAME = "LightWeightModel_ExtraExtraLight_V2_rerun"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CSV = "supervised_learning/training_data/merged_windowed_dataset.csv"
FEATURE_INFO = "supervised_learning/training_data/feature_info.json"

PROJ_OUTPUT_DIR = f"supervised_learning/output/{RUN_NAME}"
CHECKPOINT_DIR = f"supervised_learning/output/{RUN_NAME}/checkpoints"

LOG_FILE = PROJ_OUTPUT_DIR + f"/traininglog.txt"

PLOT_SAVE_PATH = PROJ_OUTPUT_DIR + f"/trainingresults_plot.png"

EPOCHS = 50000
BATCH_SIZE = 4
LR = 1e-3
SEED = 42

STEP_SIZE = 1000  # for LR scheduler
GAMMA = 0.5  # for LR scheduler

patience = 5000
no_improve = 0

# Clear checkpoint folder if it exists
if os.path.exists(CHECKPOINT_DIR):
    for file in os.listdir(CHECKPOINT_DIR):
        file_path = os.path.join(CHECKPOINT_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Redirect *all* stdout to the file only
sys.stdout = open(LOG_FILE, "w", encoding="utf-8")

print(f"=== Training Log Started: {datetime.now()} ===")
print(f"Run Name: {RUN_NAME}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print(f"Data Source: {DATA_CSV}")
print(f"Feature Info: {FEATURE_INFO}")
print(f"Output Directory: {PROJ_OUTPUT_DIR}")
print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
print(f"Plot Save Path: {PLOT_SAVE_PATH}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LR}")
print(f"LR Scheduler: Step Size = {STEP_SIZE}, Gamma = {GAMMA}")
print("="*60)

torch.manual_seed(SEED)
np.random.seed(SEED)


def plot_training_curves():
        
    # === Plot training curves ===
    epochs_range = np.arange(1, len(train_losses_hist) + 1)
    plt.figure(figsize=(12, 10), dpi=300)

    # RMSE plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses_hist, label='Train RMSE', color='blue',linewidth=0.8)
    plt.plot(epochs_range, val_losses_hist, label='Val RMSE', color='orange',linewidth=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (kWh)')
    plt.yscale('log')
    plt.title('Training & Validation RMSE')
    plt.grid(alpha=0.3)
    plt.legend()

    # R² plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_r2_hist, label='Train R²', color='green',linewidth=0.8)
    plt.plot(epochs_range, val_r2_hist, label='Val R²', color='red',linewidth=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Training & Validation R²')
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    
    return None

# === Load data ===
train_loader, val_loader, test_loader = get_loaders_from_files(
    merged_csv_path=DATA_CSV,
    feature_info_path=FEATURE_INFO,
    sequence_length=24,
    batch_size=BATCH_SIZE,
    num_workers=2,
)

# === Initialize model ===
model = TemporalAttentiveFusionNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

print(f"\n✅ Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.\n")
print(model)
print("StepLR Scheduler: step_size =", STEP_SIZE, ", gamma =", GAMMA)
print("="*60)
print("Starting training...\n")
# === TensorBoard ===
writer = SummaryWriter(log_dir="runs/TemporalFusionNet")

# === Training loop ===
best_val_loss = float("inf")
save_path = None

train_losses_hist, val_losses_hist = [], []
train_mae_hist, val_mae_hist = [], []
train_r2_hist, val_r2_hist = [], []



for epoch in range(EPOCHS):
    model.train()
    train_losses, train_preds, train_labels = [], [], []

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_preds.extend(out.detach().cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    train_r2 = r2_score(train_labels, train_preds)
    # train_losses stores MSE per batch; convert to RMSE for reporting
    avg_train_loss = np.sqrt(np.mean(train_losses))

    # === Validation ===
    model.eval()
    val_losses, val_preds, val_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            val_losses.append(loss.item())
            val_preds.extend(out.cpu().numpy())
            val_labels.extend(y.cpu().numpy())
            

    # val_losses are MSE values per batch -> convert to RMSE
    avg_val_loss = np.sqrt(np.mean(val_losses))
    val_r2 = r2_score(val_labels, val_preds)

    scheduler.step()

    # === Record history ===
    train_losses_hist.append(avg_train_loss)
    val_losses_hist.append(avg_val_loss)
    train_r2_hist.append(train_r2)
    val_r2_hist.append(val_r2)
    train_mae_hist.append(np.mean(np.abs(np.array(train_labels) - np.array(train_preds))))
    val_mae_hist.append(np.mean(np.abs(np.array(val_labels) - np.array(val_preds))))
    
    # === Logging ===
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train RMSE: {avg_train_loss:.5f} kWh | Val RMSE: {avg_val_loss:.5f} kWh | "
          f"Train R²: {train_r2:.3f} | Val R²: {val_r2:.3f} | "
          f"Patience: {no_improve}/{patience}")

    # Log RMSE to TensorBoard (kWh)
    writer.add_scalars("RMSE_kWh", {"Train": avg_train_loss, "Val": avg_val_loss}, epoch)
    writer.add_scalars("R2", {"Train": train_r2, "Val": val_r2}, epoch)

    # === Save best model ===
    if avg_val_loss < best_val_loss:
        no_improve = 0
        best_val_loss = avg_val_loss
        # Save model if both RMSE improved and R² is acceptable
        if val_r2 > 0.5:
            save_path = os.path.join(CHECKPOINT_DIR, f"best_model_{avg_val_loss:.3f}_r2_{val_r2:.3f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"\n✅ Saved best model to {save_path}\n")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            plot_training_curves()
            break
        elif np.array(val_r2_hist[-50:]).std() < 0.001:
            print(f"\nEarly stopping triggered due to R² stagnation over last 50 epochs.")
            plot_training_curves()
            break
        
    if epoch % 50 == 0:
        plot_training_curves()
        np.save(f"{PROJ_OUTPUT_DIR}/train_losses_hist.npy", np.array(train_losses_hist))
        np.save(f"{PROJ_OUTPUT_DIR}/val_losses_hist.npy", np.array(val_losses_hist))
        np.save(f"{PROJ_OUTPUT_DIR}/train_r2_hist.npy", np.array(train_r2_hist))
        np.save(f"{PROJ_OUTPUT_DIR}/val_r2_hist.npy", np.array(val_r2_hist))
        np.save(f"{PROJ_OUTPUT_DIR}/train_mae_hist.npy", np.array(train_mae_hist))
        np.save(f"{PROJ_OUTPUT_DIR}/val_mae_hist.npy", np.array(val_mae_hist))

writer.close()

# === Final Test Evaluation ===
if save_path is not None:
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        out = model(x)
        test_preds.extend(out.cpu().numpy())
        test_labels.extend(y.cpu().numpy())

test_r2 = r2_score(test_labels, test_preds)
test_mae = np.mean(np.abs(np.array(test_labels) - np.array(test_preds)))
# Compute RMSE on test set
test_mse = np.mean((np.array(test_labels) - np.array(test_preds))**2)
test_rmse = np.sqrt(test_mse)
print(f"\n✅ Test R²: {test_r2:.4f}, MAE: {test_mae:.4f} kWh, RMSE: {test_rmse:.4f} kWh")
