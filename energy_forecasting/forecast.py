# ==============================================================================
# Desc : Forecasts the energy load in the next 1 or 3 hours
# Usage : To run from terminal enter, python forecast.py --hours 3
# Notes
# - if the model was trained on x number of features, the forecasting dataset must contain 
#   x number of features. i.e pay attention to the columns day_of_week that might not
#   contain all days, in which case the one hot encoding will only generate features
#   for the available number of days
# ==============================================================================
import numpy as np
import pandas as pd

import torch 
from model.models import LSTM 
from utils.utils import generate_cyclic_features,ohe, windowed_dataset

from sklearn.compose import make_column_transformer 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import json
from termcolor import colored

def ohe_train(df:pd.DataFrame,columns:list[str]):
    """
    Used to train the one hot encoder on training set, as testing set might not contain all features
    Inputs
        - df : training dataset
        - columns : columns to be one hot encoded
    Outputs
        - ohe_encoder : trained encoder, not transformation has been done
    """
    column_trans = make_column_transformer(
        (OneHotEncoder(), columns),
        remainder = "passthrough"
    )
    ohe_encoder = column_trans.fit(df)
    ohe_arr = ohe_encoder.transform(df)

    return ohe_arr, ohe_encoder


if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Load Data to be forecasted
    # --------------------------------------------------------------------------
    # lets get the last 100 points
    df = pd.read_csv("data/load_testset.csv")

    # --------------------------------------------------------------------------
    # Load Best Hyperparameters
    # --------------------------------------------------------------------------
    with open("model/hyperparams.json", "r") as f:
        hyperparams = json.load(f)


    # --------------------------------------------------------------------------
    # Loading Training Dataset
    #   - why? : Cause the One Hot Encoder needs to be retrained to capture all days 
    #           of week
    # --------------------------------------------------------------------------
    df_train = pd.read_csv("data/load_ammended.csv")
    #df_train[-48:].to_csv("data/load_testset.csv")


    # --------------------------------------------------------------------------
    # Feature Transformation
    # --------------------------------------------------------------------------
    # One hot encode Day Of Week
    # OHE must be trained on the training set to ensure all features are included (sun, mon, ..., sat)
    _,ohe_encoder = ohe_train(df_train.drop(["hour","energy_load"], axis = 1), ["day_of_week"])

    ohe_test_arr = ohe_encoder.transform(df.drop("hour", axis = 1))
    assert ohe_test_arr.shape[1] == 7, "There should be 8 features, energy_load, dow_0, dow_1, dow_6"

    # Generate sin_hour, cos_hour : Replace hour with these components as NN will inherently learn better.
    df = generate_cyclic_features(df, "hour", 24)
    
    # Windowed Dataset
    seq, y = windowed_dataset(seq = df["energy_load"], ws = 24)
    normaliser = MinMaxScaler()


    last_seq = normaliser.fit_transform(seq)
    #last_seq = seq
    cyclic_seq = df[["sin_hour","cos_hour"]].values[-hyperparams["window_size"]:]
    ohe_seq = ohe_test_arr.toarray()[-hyperparams["window_size"]:]

    assert last_seq.shape[0] == cyclic_seq.shape[0] == ohe_seq.shape[0], "Number of values in features are not equal"
    
    #print(last_seq.shape, cyclic_seq.shape, ohe_seq.shape)

    
    X = np.hstack((last_seq,ohe_seq,cyclic_seq))
    
    
    # ==========================================================================
    # Torch Tensor + Batching
    # ==========================================================================

    X = torch.Tensor(X).unsqueeze(0) # convert to tensor and add batch dimension at 0 idx
    #print(X.shape) #N,L,H_in


    # --------------------------------------------------------------------------
    # Reconstruct Model + Load weight and biases
    # --------------------------------------------------------------------------
    
    if hyperparams["model"] == "lstm" or "gru":

        models = {
            "lstm" : LSTM,
            #todo GRU needs to be added
        }
        
        model = models[hyperparams["model"]](
            input_size = X.shape[2],
            hidden_size= hyperparams["hidden_size"],
            num_layers = hyperparams["num_layers"],
            output_size = 1,
            dropout_prob = 0.2
        )

    # Load weights and biases
    model.load_state_dict(torch.load("model/state_dict/lstm.pt"))
    model.eval()
    # --------------------------------------------------------------------------
    # Forecast values - next hour
    # --------------------------------------------------------------------------
    yhat_fsthr, hn = model.forward(X)

    ws = hyperparams["window_size"]

    zero_vect = torch.zeros((ws,ws))
    zero_vect[ws - 1,ws - 1] = yhat_fsthr.item()
    yhat_fsthr_unnorm = normaliser.inverse_transform(zero_vect.detach().numpy())

    print(f"Forecast of next hour : {yhat_fsthr_unnorm[ws - 1,ws - 1]}")

    # --------------------------------------------------------------------------
    # Forecast values - next 3 hours
    # --------------------------------------------------------------------------

