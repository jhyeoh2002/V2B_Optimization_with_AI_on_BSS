import sys
import os
import warnings
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm import tqdm

# Local Modules
import config as cfg
from supervised_learning.dataloader.loader import get_loaders_from_files
from supervised_learning.dataloader.preprocess import merge_and_process
from supervised_learning.models.STAF_V3 import TemporalAttentiveFusionNet
from util.case_dir import case_dir

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_configured_model(device):
    """
    Initializes the model with the specific anti-overfitting parameters 
    discussed (coarser bins, smaller embeddings, higher dropout).
    """
    model = TemporalAttentiveFusionNet(
        num_embeddings=500,     # Reduced from 10,000 to prevent memorization
        embedding_dim=16,       # Reduced from 64 to reduce parameter count
        n_heads=4,
        fc_hidden_dim1=64,      # Reduced MLP size
        fc_hidden_dim2=8,
        dropout=0.4,            # Increased dropout
        attention_dropout=0.2
    )
    return model.to(device)

def plot_training_curves(train_losses, val_losses, train_r2, val_r2, save_path):
    """Generates and saves training history plots."""
    epochs_range = np.arange(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 10), dpi=300)

    # Subplot 1: RMSE
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, train_losses, label='Train RMSE', color='blue', linewidth=0.8)
    plt.plot(epochs_range, val_losses, label='Val RMSE', color='orange', linewidth=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (kWh)')
    plt.title('Training & Validation RMSE')
    plt.grid(alpha=0.3)
    plt.legend()

    # Subplot 2: R2
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_r2, label='Train R²', color='green', linewidth=0.8)
    plt.plot(epochs_range, val_r2, label='Val R²', color='red', linewidth=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Training & Validation R²')
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def setup_directories(checkpoint_dir):
    """Cleans up old checkpoints or creates directory if missing."""
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train(case_id, tolerance):
    
    # --- 1. Setup & Config ---
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Indentation string for logging
    p = "\t\t"

    # Hyperparameters
    EPOCHS = 1000
    BATCH_SIZE = 4
    LR = 5e-2
    SEED = 42
    GAMMA = 0.25
    WEIGHT_DECAY = 5e-5
    LR_STEP_PATIENCE = 15
    EARLY_STOPPING_PATIENCE = LR_STEP_PATIENCE * 10
    
    # Paths & Identifiers
    RUN_NAME = f"STAFV2_{tolerance}_CASE{case_id}"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BASE_OP_FOLDER = f"./data/optimization_results"
    CASE_OP_FOLDER = case_dir(BASE_OP_FOLDER, case_id)
    
    DATA_CSV = f"./data/training_results/merged_windowed_dataset_{RUN_NAME}.csv"
    FEATURE_INFO = f"./data/training_results/feature_info_{RUN_NAME}.json"
    PROJ_OUTPUT_DIR = f"./data/training_results/{RUN_NAME}"
    CHECKPOINT_DIR = os.path.join(PROJ_OUTPUT_DIR, "checkpoints")
    PLOT_SAVE_PATH = os.path.join(PROJ_OUTPUT_DIR, "trainingresults_plot.png")

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Directory Prep
    setup_directories(CHECKPOINT_DIR)

    # --- 2. Logging Header ---
    print(f"{p}=== Training Log Started: {datetime.now()} ===")
    print(f"{p}Run Name: {RUN_NAME}")
    print(f"{p}Device: {DEVICE}")
    print(f"{p}Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    print(f"{p}" + "="*60)

    # --- 3. Data Processing ---
    merge_and_process(
        sequence_length=24,
        save_feature_info=True,
        tolerance=tolerance,
        window_size=cfg.WINDOW_SIZE,
        optimization_folder=CASE_OP_FOLDER,
        dataset_name=DATA_CSV,
        feature_info_name=FEATURE_INFO,
        case_id=case_id
    )

    train_loader, val_loader = get_loaders_from_files(
        merged_csv_path=DATA_CSV,
        feature_info_path=FEATURE_INFO,
        sequence_length=24,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    # --- 4. Model & Optimizer Init ---
    model = get_configured_model(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=GAMMA, patience=LR_STEP_PATIENCE, 
        min_lr=1e-7, verbose=False
    )

    print(f"\n{p}✅ Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} params.")
    print(f"{p}" + "="*60)

    # Print Table Header
    header_str = (f"{'Epoch':^10} | {'LR':^10} | {'Train RMSE':^12} | "
                  f"{'Val RMSE':^12} | {'Train R²':^10} | {'Val R²':^10} | {'Patience':^10}")
    print(f"{p}{'-'*len(header_str)}")
    print(f"{p}{header_str}")
    print(f"{p}{'-'*len(header_str)}")

    # --- 5. Training Loop ---
    best_val_loss = float("inf")
    no_improve = 0
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [],
        'train_mae': [], 'val_mae': []
    }

    for epoch in range(EPOCHS):
        
        # --- A. Training Step ---
        model.train()
        train_step_losses, train_preds, train_labels = [], [], []

        for x, y in train_loader:
            if torch.isnan(y).any() or torch.isnan(x).any():
                continue
            
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_step_losses.append(loss.item())
            train_preds.extend(out.detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        avg_train_loss = np.sqrt(np.mean(train_step_losses))
        train_r2 = r2_score(train_labels, train_preds)

        # --- B. Validation Step ---
        model.eval()
        val_step_losses, val_preds, val_labels = [], [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                out = model(x)
                loss = criterion(out, y)
                val_step_losses.append(loss.item())
                val_preds.extend(out.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        avg_val_loss = np.sqrt(np.mean(val_step_losses))
        val_r2 = r2_score(val_labels, val_preds)

        # Update Scheduler
        scheduler.step(val_r2)
        current_lr = optimizer.param_groups[0]["lr"]

        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        # --- C. Logging ---
        print(f"{p}{epoch+1:>5d}/{EPOCHS:<4} | "
              f"{current_lr:<10.6f} | "
              f"{avg_train_loss:>10.5f}   | "
              f"{avg_val_loss:>10.5f}   | "
              f"{train_r2:>10.3f} | "
              f"{val_r2:>10.3f} | "
              f"{no_improve:>4d}/{EARLY_STOPPING_PATIENCE:<5}")

        # --- D. Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            no_improve = 0
            best_val_loss = avg_val_loss
            
            # Only save if R2 is reasonable
            if val_r2 > 0.4:
                save_path = os.path.join(CHECKPOINT_DIR, f"best_model_{avg_val_loss:.3f}_r2_{val_r2:.3f}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"{p}{'':<41} ╚══> ✅ SAVED (RMSE: {avg_val_loss:.4f})")
        else:
            no_improve += 1
            
            # Check Stop Conditions
            stop_condition_1 = no_improve >= EARLY_STOPPING_PATIENCE
            stop_condition_2 = (len(history['val_r2']) > 50 and 
                                np.array(history['val_r2'][-50:]).std() < 0.001)
            
            if stop_condition_1 or stop_condition_2:
                msg = "Early stopping triggered." if stop_condition_1 else "Early stopping triggered (R² stagnation)."
                print(f"\n{p}{msg}")
                plot_training_curves(history['train_loss'], history['val_loss'], 
                                     history['train_r2'], history['val_r2'], PLOT_SAVE_PATH)
                break

        # Periodic Plotting
        if epoch % 5 == 4:
            plot_training_curves(history['train_loss'], history['val_loss'], 
                                 history['train_r2'], history['val_r2'], PLOT_SAVE_PATH)