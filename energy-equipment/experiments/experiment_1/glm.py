import os
os.chdir("experiments/experiment_1")

from termcolor import colored

from utils import load_data, get_cv_result, eval_metrics

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy import stats
import numpy as np

import mlflow 
import mlflow.sklearn

def train_model(param_grid:dict, train_set:tuple):
    
    # Get X_train, y_train
    X_train, y_train = train_set

    # Construct pipeline
    pipe = Pipeline([
        #("scaler",MinMaxScaler()),
        ("glm",ElasticNet())
    ])

    # Randomized Search
    search = RandomizedSearchCV(
        pipe,
        param_grid,
        scoring = ("neg_mean_squared_error"),
        cv = KFold(n_splits = 5, shuffle = True),
        n_jobs = -1,
        refit = True,
        n_iter = 100,
        verbose = 2,
        return_train_score = True
    )
    
    # Fit model
    search.fit(X_train, y_train)

    return search

if __name__ == "__main__":

    # Start MLFlow
    mlflow.set_experiment("fouling thickness")

    # Get Data
    df = load_data("../../data/Dataset1_prediction.xlsx")
    print(df.head())

    X = df.drop(["f_thickness1", "f_thickness2"], axis = 1).values 
    y = df[["f_thickness1", "f_thickness2"]].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    #print(f"\nX_train : {X_train.shape}, y_train : {y_train.shape}")
    #print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    # --------------------------------------------------------------------------
    # Base Model
    # --------------------------------------------------------------------------

    #glm = ElasticNet()
    #glm.fit(X_train, y_train)
    #print(glm.predict(X_test))

    # Accuracy of Base Model
    #f_t1 - mse : 3.2360022379493767, mae : 1.512705147031476
    #f_t2 - mse : 1.4824731200533168, mae : 1.0005220758767044
    #avg peformance (f_t1 & ft_2) - mse : 2.3592376790013456, mae : 1.2566136114540907

    # --------------------------------------------------------------------------
    # Tuned Model
    # --------------------------------------------------------------------------
    # Define parameter grid
    
    param_grid = {
        "glm__alpha" : stats.uniform(1e-5,100),
        "glm__l1_ratio" : np.arange(0,1,0.01),
    }

    # Train Model
    search = train_model(param_grid, (X_train, y_train))

    print(f"Best Params : {search.best_params_}")
    print(f"Best train score : {search.best_score_}")

    # Cv results
    results, p = get_cv_result(search)
    results = results[[
        "param_glm__alpha",
        "param_glm__l1_ratio",
        "mean_train_score",
        "std_train_score"
    ]]

    print(results.head())
    p.show()
    
    # --------------------------------------------------------------------------
    # Performance Evaluation on testset
    # --------------------------------------------------------------------------

    # Check performance Metrics
    performance_measures = eval_metrics(search, (X_test, y_test))
    
    
    with mlflow.start_run():
        mlflow.log_param("alpha", search.best_params_["glm__alpha"])
        mlflow.log_param("l1_ratio", search.best_params_["glm__l1_ratio"])
        mlflow.log_metric("avg_train_mse", search.best_score_)
        mlflow.log_metric("avg_test_mse", performance_measures["ft_avg_mse"])
        mlflow.log_metric("avg_test_mae", performance_measures["ft_avg_mae"])
        mlflow.log_metric("ft1_test_mse", performance_measures["ft1_mse"])
        mlflow.log_metric("ft1_test_mae", performance_measures["ft1_mae"])
        mlflow.log_metric("ft2_test_mse", performance_measures["ft2_mse"])
        mlflow.log_metric("ft2_test_mae", performance_measures["ft2_mae"])

        mlflow.sklearn.log_model(search, "models/glm_model")
    