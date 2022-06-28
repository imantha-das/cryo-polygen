import numpy as np
import pandas as pd

import torch 
from model.models import LSTM 
from utils.utils import generate_cyclic_features,ohe, windowed_dataset

import json
from termcolor import colored

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Load Best Hyperparametr
    # --------------------------------------------------------------------------
    with open("model/hyperparams.json", "r") as f:
        hyperparams = json.load(f)

    # --------------------------------------------------------------------------
    # Predicting features
    # --------------------------------------------------------------------------
    df = pd.read_csv("data/load_ammended.csv")
    # lets get the last 100 points
    df = df[-48:]
    print(colored(df.shape, "green"))

    # --------------------------------------------------------------------------
    # Feature Transformation
    # --------------------------------------------------------------------------
    # Generate sin_hour, cos_hour
    # Replace hour with these components as NN will inherently learn better.
    df = generate_cyclic_features(df, "hour", 24)

    # One hot encode day_of_week
    ohe_arr = ohe(df, ["day_of_week"])

    print(df.head())

    # Windowed dataset - removes last incomplete window
    X,y = windowed_dataset(seq = df["energy_load"], ws= 24)

    # Remove the last incompleted window (Since the windowed dataset removes incompleted window)
    ohe_arr = ohe_arr[:len(X)]

    # Stack features
    X = np.hstack((X, ohe_arr))
    print(X.shape)
    

    # if hyperparams["model"] == "lstm" or "gru":

    #     models = {
    #         "lstm" : LSTM,
    #         #todo GRU needs to be added
    #     }
        
    #     model = models[hyperparams["model"]](
    #         input_size = X.shape[1],
    #         hidden_size= hyperparams["hidden_size"],
    #         num_layers = hyperparams["num_layers"],
    #         output_size = 1,
    #         dropout_prob = 0.2
    #     )

    # torch.load()