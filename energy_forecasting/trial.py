# ==============================================================================
# Desc : Script to perform model training and hyperparam selection
# ==============================================================================
# ==============================================================================
# Imports
# ==============================================================================
import pandas as pd 
import numpy as np
from utils.utils import windowed_dataset, generate_cyclic_features, ohe, train_val_test_split, normalise, torch_dataset, inverse_normalise
from train import train,evaluate, plot_losses, plot_seq, calculate_metrics
from model.models import LSTM

# Check if you can remove this

import torch.nn as nn
import torch.optim as optim

from termcolor import colored
import plotly.graph_objects as go

import nni

#Todo

#todo// : OneHot Encode hour column 
#todo//, search.py to tune hyperparameters
#todo//, find best hyperparms
#todo, train model on best parameters 
#todo, functions to predict 1hour and 3hours --forecast.py 
#todo, TCN and hybrid LSTM --model.py, (do they perform better?)

# ==============================================================================
# Hyperparams
# ==============================================================================
# --------------------------------------------------------------------------
# Hyperparams
# --------------------------------------------------------------------------
# window_size = 24
# batch_size = 32
# val_ratio = 0.15
# test_ratio = 0.15

# epochs = 50 #75 
# model_name = "lstm"
# hidden_size = 128
# num_layers = 2
# optimizer_name = "adam"
# lr = 0.001
#weight_decay = ??
#dropout_prob = ??

# Seach parameters from NNI
search_params = nni.get_next_parameter()

model_name = search_params["model"]
hidden_size = search_params["hidden_size"]
num_layers = search_params["num_layers"]

optimizer_name = search_params["optimizer"]
lr = search_params["learning_rate"]

window_size = search_params["window_size"]
batch_size = search_params["batch_size"]

epochs = search_params["epochs"]

# Other params that you may want to include
val_ratio = 0.15
test_ratio = 0.15

# ==============================================================================
# Main function
# ==============================================================================

if __name__ == "__main__":



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
    X,y = windowed_dataset(seq = df["energy_load"], ws= window_size)

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
        batch_size = batch_size
    )

    # --------------------------------------------------------------------------
    # Model Training
    # --------------------------------------------------------------------------
    models = {"lstm" : LSTM}
    model = models[model_name](input_size = X.shape[1], hidden_size = hidden_size, num_layers = num_layers)
    optimizers = {
        "adam" : optim.Adam(model.parameters(), lr = lr),
        "adamax" : optim.Adamax(model.parameters(), lr = lr),
        "sgd" : optim.SGD(model.parameters(), lr = lr)
    }
    optimizer = optimizers[optimizer_name] #select optimizer from optimizers
    criterion = nn.MSELoss()

    training_losses, val_losses = train(model,train_loader, val_loader, criterion, optimizer, epochs)

    p1 = plot_losses(training_losses, val_losses, epochs)
    #p1.show()

    # --------------------------------------------------------------------------
    # Evaluation - Testig set
    # --------------------------------------------------------------------------

    test_predictions, test_original = evaluate(model, test_loader)
    
    # Inverse normalise preditions and originals
    test_predictions = inverse_normalise(np.array(test_predictions).flatten(), normaliser).flatten()
    test_original = inverse_normalise(np.array(test_original).flatten(), normaliser).flatten()

    p2 = go.Figure()
    p2 = plot_seq(test_original, "original", p2)
    p2 = plot_seq(test_predictions,"predictions", p2)
    #p2.show()

    # Evaluation - Validation ------------------------------------------------
    
    val_predictions, val_original = evaluate(model, val_loader)

    # Un-normalise predictions and originals
    val_predictions = inverse_normalise(np.array(val_predictions).flatten(), normaliser).flatten()
    val_original = inverse_normalise(np.array(val_original).flatten(), normaliser).flatten()

    p3 = go.Figure()
    p3 = plot_seq(val_original, "original", p3)
    p3 = plot_seq(val_predictions, "predictions", p3)
    #p3.show()

    # Evaluation - Training -------------------------------------------------
    
    train_predictions, train_original = evaluate(model, train_loader)

    # Un-normalise predictions and originals
    train_predictions = inverse_normalise(np.array(train_predictions).flatten(), normaliser).flatten()
    train_original = inverse_normalise(np.array(train_original).flatten(), normaliser).flatten()

    p4 = go.Figure()
    p4 = plot_seq(train_original, "original", p4)
    p4 = plot_seq(train_predictions, "predictions", p4)
    #p4.show()


    original_values = np.vstack((train_original.reshape(-1,1), val_original.reshape(-1,1), test_original.reshape(-1,1)))

    predicted_values = np.vstack((train_predictions.reshape(-1,1), val_predictions.reshape(-1,1), test_predictions.reshape(-1,1)))
    df_forecast = pd.DataFrame({"y" : original_values.flatten(), "yhat" : predicted_values.flatten()})
    #df_forecast.to_csv(path_or_buf = "data/forecast_values.csv")
    
    p5 = go.Figure()
    p5 = plot_seq(df_forecast["y"], "original", p5)
    p5 = plot_seq(df_forecast["yhat"], "predictions", p5)
    #p5.show()
    
    # --------------------------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------------------------

    metrics = calculate_metrics(test_original, test_predictions)
    rmse = metrics["rmse"] #rmse on test set
    
    print(f"rmse : {metrics['rmse']:.4f}\tmae : {metrics['mae']:.4f}\tr2 : {metrics['r2']:.4f}")

    # Metrics validation
    metrics_val = calculate_metrics(val_original, val_predictions)
    print(f"rmse_val : {metrics_val['rmse']:.4f}\tmae_val : {metrics_val['mae']:.4f}\tr2_val : {metrics_val['r2']:.4f}")
    
    # Metrics train
    metrics_train = calculate_metrics(train_original, train_predictions)
    print(f"rmse_train : {metrics_train['rmse']:.4f}\tmae_train : {metrics_train['mae']:.4f}\tr2_train : {metrics_train['r2']:.4f}")
    
    # Report validation metrics to nni
    nni.report_final_result(metrics_val["rmse"])
