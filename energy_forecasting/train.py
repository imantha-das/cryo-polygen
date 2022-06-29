import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.graph_objs as go

import json
from datetime import datetime

from model.models import LSTM
from utils.utils import windowed_dataset, generate_cyclic_features, ohe, train_val_test_split, normalise, torch_dataset, inverse_normalise

# ==============================================================================
# Train functions
# ==============================================================================
def train(model,train_loader:DataLoader, val_loader,criterion:nn.modules.loss,optimizer:optim,epochs:int)->tuple[list,list]:

    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for (X_batch,y_batch) in train_loader:
            # SHAPES : X - (64, 27), y - (64,1) where batch_size = 64
            # LSTM model requires, (N, L, H_in)
            X_batch = X_batch.unsqueeze(1) # Add dimension at 1, for L
            yhat,_ = model.forward(X_batch) # (64,1) ... for each iteration

            # Compute Loss
            loss = criterion(yhat, y_batch)
            # Keep track all losses in batch
            batch_losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Keep track of mean loss in batch
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        
        # Validation losses
        with torch.no_grad():
            val_batch_losses = []
            for (X_batch_val, y_batch_val) in val_loader:
                X_batch_val = X_batch_val.unsqueeze(1)
                yhat_val, _ = model.forward(X_batch_val)

                val_batch_losses.append(criterion(yhat_val, y_batch_val).item())
            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)

        if (epoch <= 10) | (epoch%50 == 0):
                print(f"[{epoch}/{epochs}] Training loss : {training_loss:.4f}\t Validation loss : {val_loss:.4f}")
    
    return training_losses,val_losses

# ------------------------------------------------------------------------------
# Evaluate 
# ------------------------------------------------------------------------------

def evaluate(model,loader:DataLoader):
    with torch.no_grad():
        predictions = []
        original_values = [] 
        for X_batch, y_batch in loader:
            # SHAPES, X_test : (64,27), y_test : (64,1)
            #SHAPES RQD by MODEL, X_test : (N, L, H_in) - (64,1,27)
            #SHAPED AFTER model.foward : (N, L, H_out)
            X_batch = X_batch.unsqueeze(1)
            model.eval()
            yhat,_ = model.forward(X_batch)
            predictions.append(yhat.detach().numpy())
            original_values.append(y_batch.detach().numpy())
        return predictions, original_values


def plot_seq(data:np.array, name:str, fig):
    """
    Desc 
    """
    fig.add_trace(go.Scatter(
        x = np.arange(0, len(data)),
        y = data, 
        name = name
    ))
    fig.update_layout(xaxis_title = "time", yaxis_title = "energy load", title = "Building Energy Load", template = "plotly_white")
    return fig

# ------------------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------------------

def plot_losses(training_losses:list, validation_losses:list, epochs:int):
    """
    Desc : Plots training and validation losses
    Inputs
        training_losses : training loss 
        validation_losses : validation loss 
        epochs : number of epochs 
    Outputs
        fig : plotly graph object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = np.arange(0, epochs),
        y = training_losses,
        name = "training loss"
    ))
    fig.add_trace(go.Scatter(
        x = np.arange(0, epochs),
        y = validation_losses,
        name = "validation loss"
    ))
    fig.update_layout(xaxis_title = "no of epochs", yaxis_title = "loss", title = "Training vs Validation loss", template = "plotly_white")
    return fig

# ------------------------------------------------------------------------------
# Compute Metrics
# ------------------------------------------------------------------------------

def calculate_metrics(original:np.array, prediction:np.array)->dict:
    """
    Desc : Computes performance metrics
    Inputs
        Original : original time series values
        prediction : predicted time series values
    Outputs
        mae : mean absolute error 
        rmse : root mean squared error 
        r2 : r2 score
    """
    return {'mae' : mean_absolute_error(original, prediction),
            'rmse' : mean_squared_error(original, prediction) ** 0.5,
            'r2' : r2_score(original, prediction)}

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Load Hyperparams
    # --------------------------------------------------------------------------
    with open("model/hyperparams.json", "r") as f:
        hyperparams = json.load(f)

    # -> {
    #       'model': 'lstm', 
    #       'hidden_size': 32, 
    #       'num_layers': 2, 
    #       'optimizer': 'sgd', 
    #       'learning_rate': 0.005, 
    #       'window_size': 24, 
    #       'batch_size': 1, 
    #       'epochs': 100
    #    }

    # other parameters (that might be added later)
    test_ratio = 0.15 
    val_ratio = 0.15
    
    # --------------------------------------------------------------------------
    # Load Dataset -> Feature Exctraction -> Train-Val-Test split -> Normalise -> Tensor Dataset
    # --------------------------------------------------------------------------

    # import data
    df = pd.read_csv("data/load_ammended.csv")

    # Generate sin_hour, cos_hour
    # Replace hour with these components as NN will inherently learn better.
    df = generate_cyclic_features(df, "hour", 24)

    # One hot encode day_of_week
    ohe_arr = ohe(df, ["day_of_week"])

    # Windowed dataset - removes last incomplete window
    X,y = windowed_dataset(seq = df["energy_load"], ws= hyperparams["window_size"])

    # Remove the last incompleted window (Since the windowed dataset removes incompleted window)
    ohe_arr = ohe_arr[:len(X)]

    # Stack features
    X = np.hstack((X, ohe_arr))

    # Train - Validation - Test Split 
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_ratio, test_ratio)

    # print("shapes") 
    # print(f"X_train : {X_train.shape}") #? (3124,27)
    # print(f"y_train : {y_train.shape}") #? (3124,)
    # print(f"X_val : {X_val.shape}") #? (552, 27)
    # print(f"y_val : {y_val.shape}") #? (552,)
    # print(f"X_test : {X_test.shape}") #? (649, 27)
    # print(f"y_test : {y_test.shape}") #? (649,)

    # Normalise Data
    (norm_data, normaliser) = normalise(X_train, X_val, X_test, y_train.reshape(-1,1), y_val.reshape(-1,1), y_test.reshape(-1,1))
    #? Note after normalising y has a shape of (*, 1)
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = norm_data

    # Get torch datasets
    train_loader, val_loader, test_loader = torch_dataset(
        X_train_norm, 
        X_val_norm, 
        X_test_norm, 
        y_train_norm, 
        y_val_norm, 
        y_test_norm,
        batch_size = hyperparams["batch_size"]
    )


    # --------------------------------------------------------------------------
    # Load Model Architecture from JSON
    # --------------------------------------------------------------------------
    
    if hyperparams["model"] == "lstm" or "gru":

        models = {
            "lstm" : LSTM,
            #todo GRU needs to be added
        }
        
        model = models[hyperparams["model"]](
            input_size = X.shape[1],
            hidden_size= hyperparams["hidden_size"],
            num_layers = hyperparams["num_layers"],
            output_size = 1,
            dropout_prob = 0.2
        )

    # --------------------------------------------------------------------------
    # Train Model
    # --------------------------------------------------------------------------
    
    lr = hyperparams["learning_rate"]

    optimizers = {
        "sdg" : optim.SGD(model.parameters(), lr),
        "adam" : optim.Adam(model.parameters(), lr),
        "adamax" : optim.Adamax(model.parameters(), lr)
    }

    epochs = hyperparams["epochs"]
    optimizer = optimizers[hyperparams["optimizer"]]
    criterion = nn.MSELoss()
    
    training_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs
    )

    # Save model
    torch.save(model.state_dict(), f"model/state_dict/{hyperparams['model']+'.pt'}")

    p1 = plot_losses(training_losses, val_losses, epochs)
    p1.show()

    # --------------------------------------------------------------------------
    # Evaluate Training Set
    # --------------------------------------------------------------------------

    test_predictions, test_original = evaluate(model, test_loader)
    
    # Inverse normalise preditions and originals
    test_predictions = inverse_normalise(np.array(test_predictions).flatten(), normaliser).flatten()
    test_original = inverse_normalise(np.array(test_original).flatten(), normaliser).flatten()

    p2 = go.Figure()
    p2 = plot_seq(test_original, "original", p2)
    p2 = plot_seq(test_predictions,"predictions", p2)
    p2.show()
    
