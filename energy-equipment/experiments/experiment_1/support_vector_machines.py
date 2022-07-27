import os
os.chdir("experiments/experiment_1")

from sklearn.svm import SVR 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy import stats
import numpy as np

import plotly.express as px

from utils import load_data, get_cv_result

import mlflow 
import mlflow.sklearn


def train_model(param_grid:dict, train_set:tuple):
    
    # Get X_train, y_train
    X_train, y_train = train_set

    # Construct pipeline
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("svm", MultiOutputRegressor(estimator = SVR()))
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
        verbose = 2,
        return_train_score = True
    )
    
    # Fit model
    search.fit(X_train, y_train)

    return search

def eval_metrics(model, test_set:tuple):

    X_test, y_test = test_set
    yhat_test = search.predict(X_test)
    mse = mean_squared_error(y_test, yhat_test)
    mae = mean_absolute_error(y_test, yhat_test)
    print(f"Root-MSE on testset : {np.sqrt(mse)}, mae : {mae}")

    return mae, mse


if __name__ == "__main__":
    # Start Mlflow - Experiment
    mlflow.set_experiment("svm (fouling thickness)")

    # Get Data
    df = load_data("../../data/Dataset1_prediction.xlsx")
    print(df.head())

    X = df.drop(["f_thickness1", "f_thickness2"], axis = 1).values 
    y = df[["f_thickness1", "f_thickness2"]].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    print(f"\nX_train : {X_train.shape}, y_train : {y_train.shape}")
    print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    

    #kf = KFold(n_splits = 5, shuffle = True)
    #cv = cross_validate(pipe, X_train, y_train, scoring = "neg_mean_squared_error", return_train_score=True)

    # --------------------------------------------------------------------------
    # Randomized Search : Training
    #   model refited to best params
    # --------------------------------------------------------------------------

    # Define parameters space
    param_grid = [
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

    search = train_model(param_grid=param_grid, train_set=(X_train, y_train))

    print(f"Best Params : {search.best_params_}")
    print(f"Best train score : {search.best_score_}")
    
    results, p = get_cv_result(search)
    print(results.head())
    p.show()


    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------
    mse, mae = eval_metrics(model = search, test_set=(X_test, y_test))
    


    with mlflow.start_run():
        mlflow.log_param("best_C", search.best_params_["svm__estimator__C"])
        mlflow.log_param("best_gamma", search.best_params_["svm__estimator__gamma"])
        mlflow.log_param("best_kernel", search.best_params_["svm__estimator__kernel"])
        mlflow.log_metric("best_train_mse", search.best_score_)
        mlflow.log_metric("best_test_mse", mse)
        mlflow.log_metric("best_test_mae", mae )

        mlflow.sklearn.log_model(search, "models/svm_model")

    


    
    



