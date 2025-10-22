import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def loaddata(dropcolumns:list):

    df = pd.read_csv("./Data/merged_Full_Data2.csv")

    # Separate features and ground truth
    X = df.drop(columns=dropcolumns)
    feature_names = X.columns.tolist()
    y = df['Charging_Results']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (important for training)
    scaler = StandardScaler()

    # Replace Inf and -Inf with NaN and drop rows with NaN in X_train and X_test
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure y_train and y_test are aligned with X_train and X_test
    y_train = y_train[X_train.index]  # Keep only the rows in y_train corresponding to cleaned X_train
    y_test = y_test[X_test.index]  # Keep only the rows in y_test corresponding to cleaned X_test

    # Reset the indices to ensure they are aligned after dropping rows
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape for compatibility
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    print(X_train_tensor.shape)
    print(y_train_tensor.shape)
    print(X_test_tensor.shape)
    print(y_test_tensor.shape)

    # Create TensorDatasets
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=1028, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1028, shuffle=False)

    # length = 24

    # for inputs, labels in train_loader:
    #     x_others = inputs[:,:-length*7]
    #     x_carbon = inputs[:,-length*7:-length*6]

    #     print(x_others)
    #     print(x_carbon)
        
    #     x_combined = torch.cat((x_others, x_carbon), dim=1)  

        # print(x_combined)  
        
        # print('\nOthers:')
        # print((inputs[0,:-length*7]))
        
        # print('\nCarbon Intensity (kgC02eq/kWh):')
        # print((inputs[0,-length*7:-length*6]))

        # print('\nRadiation Intensity, I:')
        # print((inputs[0,-length*6:-length*5]))

        # print('\nTemperature,T (deg):')
        # print((inputs[0,-length*5:-length*4]))

        # print('\nPV Generation (kWh):')
        # print((inputs[0,-length*4:-length*3]))

        # print('\nBuilding Electricity Cost (NTD/kWh):')
        # print((inputs[0,-length*3:-length*2]))
        
        # print('\nVehicle Electricity Cost (NTD/kWh):')
        # print((inputs[0,-length*2:-length]))
        
        # print("\nEnergy:")
        # print((inputs[0,-24:]))
        # break
    
    return train_loader, test_loader
