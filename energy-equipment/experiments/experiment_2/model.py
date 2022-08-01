# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
import sys
import numpy as np 
import pandas as pd 

import plotly.express as px 

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from scipy import stats 

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


# ------------------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------------------

def get_data(path):
    df = pd.read_csv(data_path)
    # Remove NaN columns
    df.drop(["Unnamed: 4", "Unnamed: 5"], axis = 1, inplace = True)
    # Remove any missing values
    df.dropna(inplace=True)
    return df

# ------------------------------------------------------------------------------
# Train Model
# ------------------------------------------------------------------------------

def train_model(param_grid, estimator,estimator_name,train_set):
    # MultiOutputRegressor(estimator = SVR())
    # Get X_train, y_train
    X_train, y_train = train_set

    # Construct pipeline
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        (estimator_name, estimator)
    ])

    # Randomized Search
    search = RandomizedSearchCV(
        pipe,
        param_grid,
        scoring = ("neg_mean_squared_error"),
        cv = KFold(n_splits = 5, shuffle = True),
        n_jobs = -1,
        refit = True,
        n_iter = 60,
        verbose = 0,
        return_train_score = True
    )
    
    # Fit model
    search.fit(X_train, y_train)
    return search

# ------------------------------------------------------------------------------
# CV Results
# ------------------------------------------------------------------------------

def get_cv_result(model):
    results = pd.DataFrame(model.cv_results_)
    results.sort_values(by = "mean_test_score", ascending=False, inplace=True)
    
    p = px.scatter(
        x = np.arange(0,results.shape[0]), 
        y = results["mean_test_score"],
        error_y=results["std_test_score"]
    )
    p.update_layout(
        template = "plotly_white", 
        xaxis_title = "hyperparam_idx", 
        yaxis_title = "MSE"
    )

    return results, p

# ------------------------------------------------------------------------------
# Prediction and Scores
# ------------------------------------------------------------------------------

def get_predictions_and_scores(model,model_type,predictions, metrics):
    yhat = model.predict(X)
    predictions[[f"yts_hat_{model_type}",f"ytt_hat_{model_type}"]] = yhat

    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae = mean_absolute_error(y, yhat)
    r2 = r2_score(y, yhat)
    var = np.var(y - yhat)

    model_metrics = pd.DataFrame({
        "model" : [model_type],
        "rmse" : [rmse],
        "mae" : [mae],
        "r2" : [r2],
        "var" : [np.var(y - yhat)]
    })
    metrics = pd.concat([metrics, model_metrics],axis = 0)

    return predictions, metrics



if __name__ == "__main__":
    # Get Data
    data_path = os.path.join("data","data1_280722.csv")
    df = get_data(data_path)

    # Train Test Split
    X = df.drop(["yts","ytt"],axis = 1).values
    y = df[["yts","ytt"]].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    #print(f"X_train : {X_train.shape}, y_train : {y_train.shape}")
    #print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    # Predictions and metrics
    predictions = pd.DataFrame(columns = ["yts", "ytt"])
    predictions[["yts","ytt"]] = y

    metrics = pd.DataFrame(columns = ["model","rmse", "mae", "r2", "var"])

    # SVM
    # Define parameters space
    param_grid_svm = [
        {
            "svm__estimator__C" : stats.uniform(0.1,100), 
            "svm__estimator__kernel" : ["linear"]
        },
        {
            "svm__estimator__C" : stats.loguniform(0.1, 100), 
            "svm__estimator__gamma" : stats.loguniform(0.0001, 10), 
            "svm__estimator__kernel" : ["poly","rbf"]
        }
    ]

    search_svm = train_model(
        param_grid=param_grid_svm, 
        estimator = MultiOutputRegressor(estimator = SVR()),
        estimator_name = "svm",
        train_set=(X_train, y_train)
    )

    print(f"Best Params : {search_svm.best_params_}")
    print(f"Best train score : {search_svm.best_score_}")

    cv_results, p = get_cv_result(search_svm)
    p.show()

    predictions, metrics = get_predictions_and_scores(search.best_estimator_, "svm", predictions, metrics)
    print(predictions.head())
    print(metrics.head())

    