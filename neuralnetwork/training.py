import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils
from forecast_embedded_model import Integrated_Model
from forecast_embedded_model import Integrated_Model2
from forecast_embedded_model import Integrated_Model3

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os 
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')
torch.cuda.empty_cache() 

dropcolumns = ['Charging_Results','Datetime', 'Ave_EnergyperSlot', 'Unnamed: 0', 'Hour', 'Month', 'Day', 'Hour_of_day']

train_loader, test_loader = utils.loaddata(dropcolumns)

# Model, loss function, optimizer, and scheduler
model = Integrated_Model3().to(device)
criterion = nn.MSELoss()  # Loss function (Mean Squared Error)
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)  # Learning rate decay

# Training function
def train(epochs, modelname):
    
    # Directory for model and plots
    output_folder = f'.trainresults/{modelname}/plots'
    model_folder = f'.trainresults/{modelname}/model'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(model_folder, f"model_{modelname}.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model {modelname} loaded successfully.")

    train_losses, test_losses, train_r2_scores, test_r2_scores = [], [], [], []
    train_mse_scores, test_mse_scores = [], []  # New lists for MSE
    train_rmse_scores, test_rmse_scores = [], []  # New lists for RMSE
    best_loss = float('inf')  # Track the best loss for saving the model
    
    # CSV file path for saving metrics
    metrics_file = os.path.join(output_folder, f"metrics_{modelname}.csv")

    # Write header to CSV file
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Test Loss", "Train R²", "Test R²", 
                         "Train MSE", "Test MSE", "Train RMSE", "Test RMSE", "Learning Rate"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        # Training Loop
        for inputs, labels in train_loader:
            
            # print(labels.size())
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs & labels to GPU
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Compute R² score for training
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        train_r2 = r2_score(all_labels, all_preds)
        train_r2_scores.append(train_r2)

        # Compute MSE and RMSE for training
        train_mse = mean_squared_error(all_labels, all_preds)
        train_rmse = np.sqrt(train_mse)
        train_mse_scores.append(train_mse)
        train_rmse_scores.append(train_rmse)

        # Compute average training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation Loop
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Compute test loss, R² score, MSE, and RMSE
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        test_r2 = r2_score(all_labels, all_preds)
        test_r2_scores.append(test_r2)

        test_mse = mean_squared_error(all_labels, all_preds)
        test_rmse = np.sqrt(test_mse)
        test_mse_scores.append(test_mse)
        test_rmse_scores.append(test_rmse)

        # Save model if it has the best validation loss
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)

        # Learning rate update
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Progress
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | "
              f"Train R²: {train_r2:.5f} | Test R²: {test_r2:.5f} | Train MSE: {train_mse:.5f} | "
              f"Test MSE: {test_mse:.5f} | Train RMSE: {train_rmse:.5f} | Test RMSE: {test_rmse:.5f} | "
              f"LR: {current_lr:.5f}")

        # Save metrics to CSV
        with open(metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, test_loss, train_r2, test_r2, 
                             train_mse, test_mse, train_rmse, test_rmse, current_lr])

        # Plot Training Results
        if epoch % 10 == 0 or epoch == epochs - 1:  # Save plot every 100 epochs
            epochs_range = range(1, len(train_losses) + 1)
            plt.figure(figsize=(12, 8), dpi = 300)

            # Loss Plot
            plt.subplot(2, 1, 1)
            plt.grid(True, alpha = 0.2)
            plt.plot(epochs_range, train_losses, label='Training Loss')
            plt.plot(epochs_range, test_losses, label='Testing Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # R² Score Plot
            plt.subplot(2, 1, 2)
            plt.grid(True, alpha = 0.2)
            plt.plot(epochs_range, train_r2_scores, label='Training R² Score')
            plt.plot(epochs_range, test_r2_scores, label='Test R² Score')
            plt.xlabel('Epochs')
            plt.ylabel('R² Score')
            plt.legend()

            plt.tight_layout()
            plot_filename = os.path.join(output_folder, f"training_results_{modelname}.png")
            plt.savefig(plot_filename)
            plt.close()

    return train_losses, test_losses, test_r2_scores, test_mse_scores, test_rmse_scores

# Entry point of script
if __name__ == "__main__":
    
    epochs = 10000
    modelname = "test_integratedmodel4"
    
    train(epochs=epochs, modelname=modelname)