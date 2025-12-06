import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score

# --- Configuration Import ---
import config as cfg

# --- Local Module Imports ---
from util.case_dir import case_dir 
from supervised_learning.utils.io import ensure_dir
from supervised_learning.utils.visualization import plot_training_curves
from supervised_learning.data_pipeline import merge_and_process, get_dataloaders
from supervised_learning.models.staf_net import TemporalAttentiveFusionNet

def train(case_id, tolerance=cfg.TOLERANCE, run_name=None, rerun = False):
    """
    Main training loop for STAF-Net.
    """
    # --- 1. Setup & Directories ---    
    # Construct Paths using config base dirs
    opt_folder = case_dir(cfg.BASE_OPT_FOLDER, case_id)
    output_dir = os.path.join(cfg.TRAIN_RESULTS_DIR, run_name)
    figure_dir = os.path.join(cfg.TRAIN_FIGURE_DIR, run_name)
    
    paths = {
        "opt_folder": opt_folder,
        "data_csv": os.path.join(output_dir, f"merged_dataset_{run_name}.csv"),
        "feature_info": os.path.join(output_dir, f"feature_info_{run_name}.json"),
        "output_dir": output_dir,
        "scaler_dir": os.path.join(output_dir, "scalers"),
        "checkpoints": os.path.join(output_dir, "checkpoints"),
        "history": os.path.join(output_dir, "history"),
        "plot": os.path.join(figure_dir, "training_plot.png")
    }
    
    
    if os.path.exists(os.path.join(paths["checkpoints"], "best_model.pth")) and not rerun:
        print(f"\t\t[INFO] Found existing model checkpoint at {paths['checkpoints']}. Skipping training as rerun is False.")
        return
    
    # Create necessary folders
    ensure_dir(paths["checkpoints"], clean=True)
    ensure_dir(paths["history"], clean=True)
    ensure_dir(paths["scaler_dir"], clean=False)
    ensure_dir(figure_dir, clean=True)
    
    # Reproducibility
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    print(f"\n=== Starting Training Run: {run_name} ===")
    print(f"Device: {cfg.DEVICE}")

    # --- 2. Data Pipeline (Preprocess & Load) ---
    
    # A. Create Dataset (CSV)
    # We pass cfg.WINDOW_SIZE to ensure consistency with your optimization logic
    merge_and_process(
        sequence_length=cfg.SEQUENCE_LENGTH,
        dataset_name=paths["data_csv"],
        feature_info_name=paths["feature_info"],
        case_id=case_id
    )

    # B. Read Metadata to Configure Model Dimensions
    with open(paths["feature_info"], "r") as f:
        feat_info = json.load(f)
    
    # Calculate exact input sizes based on the processed data
    dims = {
        "static": len(feat_info["static_cols"]),
        "series_count": len(feat_info["series_blocks"]),
        "seq_len": len(feat_info["series_blocks"][0]),
        "vehicle": sum(len(block) for block in feat_info["battery_blocks"]),
        # Fallback to config default if not in json
        "emb_vocab": cfg.NUM_EMBEDDINGS
    }
    print(f"\t[Model Config] Detected Dims: {dims}")

    # C. Create DataLoaders
    train_loader, val_loader = get_dataloaders(
        merged_csv_path=paths["data_csv"],
        feature_info_path=paths["feature_info"],
        sequence_length=dims["seq_len"],
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        scaler_dir=paths["scaler_dir"],
        fit_scaler=True,
        random_seed=cfg.SEED
    )

    # --- 3. Model Initialization ---
    model = TemporalAttentiveFusionNet(
        num_static=dims["static"],
        num_series=dims["series_count"],
        sequence_length=dims["seq_len"],
        vehicle_input_dim=dims["vehicle"],
        
        # Hyperparameters from Config
        num_embeddings=dims["emb_vocab"],
        embedding_dim=cfg.EMBEDDING_DIM,
        n_heads=cfg.N_HEADS,
        fc_hidden_dim1=cfg.HIDDEN_DIM_1,
        fc_hidden_dim2=cfg.HIDDEN_DIM_2,
        dropout=cfg.DROPOUT,
        attention_dropout=cfg.ATTENTION_DROPOUT
    ).to(cfg.DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    
    # Scheduler: Reduces LR when validation R2 stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=cfg.GAMMA, patience=cfg.LR_PATIENCE
    )
    # --- MODEL SUMMARY LOGGING ---
    # Calculate trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\t\t[INFO] \n" + "="*50)
    print(f"\t\t[INFO] 🚀 TRAINING SESSION: {run_name}")
    print("\t\t[INFO] "+"="*50)
    print(f"\t\t[INFO] Device:             {cfg.DEVICE}")
    print(f"\t\t[INFO] Model Parameters:   {num_params:,}") # The comma adds thousands separators (e.g. 1,050,200)
    print("\t\t[INFO] "+"-" * 50)
    print(f"\t\t[INFO] Epochs:             {cfg.EPOCHS}")
    print(f"\t\t[INFO] Batch Size:         {cfg.BATCH_SIZE}")
    print(f"\t\t[INFO] Learning Rate:      {cfg.LEARNING_RATE}")
    print(f"\t\t[INFO] Weight Decay:       {cfg.WEIGHT_DECAY}")
    print(f"\t\t[INFO] Dropout:            {cfg.DROPOUT}")
    print("\t\t[INFO] "+"-" * 50)
    print(f"\t\t[INFO] Hidden Layers:      [{cfg.HIDDEN_DIM_1}, {cfg.HIDDEN_DIM_2}]")
    print(f"\t\t[INFO] Embedding Dim:      {cfg.EMBEDDING_DIM}")
    print(f"\t\t[INFO] Attention Heads:    {cfg.N_HEADS}")
    print("\t\t[INFO]"+"="*50 + "\n")
    # --- 4. Training Loop ---
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
    best_val_loss = float("inf")
    no_improve = 0
    
    print("\n\t\t[INFO]Epoch |    LR    | Trn RMSE | Val RMSE | Trn R2 | Val R2 | Stop")
    print("-" * 65)

    for epoch in range(cfg.EPOCHS):
        # A. Train
        model.train()
        t_loss, t_preds, t_targets = [], [], []
        
        for x, y in train_loader:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            t_loss.append(loss.item())
            t_preds.extend(out.detach().cpu().numpy())
            t_targets.extend(y.cpu().numpy())
            
        train_rmse = np.sqrt(np.mean(t_loss))
        train_r2 = r2_score(t_targets, t_preds)

        # B. Validate
        model.eval()
        v_loss, v_preds, v_targets = [], [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE).unsqueeze(1)
                out = model(x)
                loss = criterion(out, y)
                v_loss.append(loss.item())
                v_preds.extend(out.cpu().numpy())
                v_targets.extend(y.cpu().numpy())

        val_rmse = np.sqrt(np.mean(v_loss))
        val_r2 = r2_score(v_targets, v_preds)

        # C. Updates & Logging
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_r2)
        
        history['train_loss'].append(train_rmse)
        history['val_loss'].append(val_rmse)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        print(f"\t\t[INFO]{epoch+1:5d} | {current_lr:.6f} | {train_rmse:8.4f} | {val_rmse:8.4f} | "
              f"{train_r2:6.3f} | {val_r2:6.3f} | {no_improve}/{cfg.ES_PATIENCE}")

        # D. Checkpointing & Early Stopping
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(paths["checkpoints"], "best_model.pth"))
            # Save history as .npy files
            for key, value in history.items():
                np.save(os.path.join(paths["history"], f"{key}.npy"), np.array(value))
        else:
            no_improve += 1
            if no_improve >= cfg.ES_PATIENCE:
                print(f"\n[Stopping] Early stopping triggered after {epoch+1} epochs.")
                break

    # --- 5. Wrap Up ---
        
        if epoch % 2 == 1:
            plot_training_curves(history, paths["plot"])
            
    print(f"Training Complete. Results saved to: {paths['output_dir']}")

if __name__ == "__main__":
    train(case_id=1)