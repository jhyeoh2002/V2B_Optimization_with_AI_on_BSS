import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history: dict, save_path: str):
    """
    Generates and saves RMSE and R2 training curves.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', etc.
        save_path (str): File path to save the image.
    """
    epochs_range = np.arange(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 7), dpi=300)

    # Subplot 1: RMSE
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train RMSE', color='green', linewidth=1.8)
    plt.plot(epochs_range, history['val_loss'], label='Val RMSE', color='red', linewidth=1.8)
    # plt.title('Training & Validation RMSE')
    plt.ylabel('RMSE (kWh)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Subplot 2: R2
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, history['train_r2'], label='Train R²', color='green', linewidth=1.8)
    plt.plot(epochs_range, history['val_r2'], label='Val R²', color='red', linewidth=1.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    # plt.title('Training & Validation R²')
    # plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
