# ==============================================================================
# Desc : Forecasts the energy load in the next 1 or 3 hours
# Usage : To run from terminal enter, python forecast.py --hours 3
#todo Currently using window_dataset which needs to be changed to accomadate for only 24 values
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

def test_feature_transformation(df_test:pd.DataFrame, df_train:pd.DataFrame, hyperparams:dict)->tuple:
    # OHE must be trained on the training set to ensure all features are included (sun, mon, ..., sat)
    _,ohe_encoder = ohe_train(df_train.drop(["hour","energy_load"], axis = 1), ["day_of_week"])
    ohe_test_arr = ohe_encoder.transform(df_test.drop("hour", axis = 1))
    assert ohe_test_arr.shape[1] == 7, "There should be 8 features, energy_load, dow_0, dow_1, dow_6"

    # Generate sin_hour, cos_hour : Replace hour with these components as NN will inherently learn better.
    df_test = generate_cyclic_features(df_test, "hour", 24)
    
    # Windowed Dataset
    seq, y = windowed_dataset(seq = df_test["energy_load"], ws = 24)
    print(colored(seq.shape, "magenta"))
    
    # Normalise test data as model was trained on normalised data
    normaliser = MinMaxScaler()
    last_seq = normaliser.fit_transform(seq)
    
    # Generate Cyclic Features
    cyclic_seq = df_test[["sin_hour","cos_hour"]].values[-hyperparams["window_size"]:]

    # One-Hot-Encode data
    ohe_seq = ohe_test_arr.toarray()[-hyperparams["window_size"]:]

    # Ensure equal number of rows are avaialble in 
    assert last_seq.shape[0] == cyclic_seq.shape[0] == ohe_seq.shape[0], "Number of values in features are not equal"
    
    #print(last_seq.shape, cyclic_seq.shape, ohe_seq.shape)

    X = np.hstack((last_seq,ohe_seq,cyclic_seq))
    X = torch.Tensor(X).unsqueeze(0) # convert to tensor and add batch dimension at 0 idx
    return X, normaliser

def forecast(model:"model.models.LSTM", X:torch.Tensor, normaliser:MinMaxScaler,hyperparams:dict)->float:
    """
    Desc : Forecast the energy load in the next period
    Inputs :
        model : Deep learning model
        X : input features 
        normaliser : MinMaxScaler
        hyperparams : Selected hyperparams during tuning phase
    Outputs :
        next forecasted value
    """

    yhat_next, hn = model.forward(X)

    ws = hyperparams["window_size"]

    yhat_seq = torch.zeros((ws,ws))
    yhat_seq[ws - 1,ws - 1] = yhat_next.item()
    yhat_next_unnorm = normaliser.inverse_transform(yhat_seq.detach().numpy())
    return yhat_next_unnorm[ws - 1,ws - 1]

def generate_features_on_forecasted_values(df, hr, next_val, hyperparams):
    """
    Desc : To forecast beyound the next value, i.e 3 hours we will require 
    day of week and hour features to be generated
    """
    # Get next hour
    prev_hour = df.hour.iloc[-1]

    if prev_hour != 23:
        next_hour = prev_hour + 1
    else:
        next_hour = 0

    # Get next day of week
    weekday_track = []
    dow_rev_ls = list(reversed(df.day_of_week.tolist()))
    last_day = dow_rev_ls[0]
    
    for i in dow_rev_ls:
        if i == last_day:
            weekday_track.append(i)
        
    if len(weekday_track) < 24:
        next_day = last_day 
    elif len(weekday_track) == 24 and last_day < 6:
        next_day = last_day + 1
    elif len(weekday_track) == 24 and last_day == 6:
        next_day = 0

    df.loc[len(df),:] = [next_day, next_hour, next_val]

    ws = hyperparams["window_size"]
    df_next_window = df.iloc[-48:,:]
    return df_next_window


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
    #df_train[-48:].to_csv("data/load_testset.csv", index = False)


    # --------------------------------------------------------------------------
    # Feature Transformation
    # --------------------------------------------------------------------------
    # One hot encode Day Of Week
    X, normaliser = test_feature_transformation(df, df_train, hyperparams)


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
    # Forecast energy demand in the next 3 hours
    # --------------------------------------------------------------------------
    yhat_all = []
    
    for hr in range(1,4):
        if hr == 1:
            yhat_next = forecast(model, X, normaliser,hyperparams)
            yhat_all.append(yhat_next)
        else:
            df_next = generate_features_on_forecasted_values(df, hr, yhat_next, hyperparams)
            X_next, normaliser = test_feature_transformation(df_next, df_train, hyperparams)
            yhat_next = forecast(model, X_next, normaliser, hyperparams)
            yhat_all.append(yhat_next)

    print(f"Forecast of next hour : {yhat_next}")
    print(f"forecasted values in the next 3 hours : {yhat_all}")

   


