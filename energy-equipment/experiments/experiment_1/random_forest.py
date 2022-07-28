import os
os.chdir("experiments/experiment_1")

from termcolor import colored

from utils import load_data, get_cv_result, eval_metrics

from sklearn.ensemble import RandomForestRegressor
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
        ("rf",RandomForestRegressor())
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

    #forest = RandomForestRegressor()
    #forest.fit(X_train, y_train)
    #print(forest.predict(X_test))

    # Base Model Performance Metrics
    #f_t1 - mse : 5.214849527289824, mae : 1.8670005724951375
    #f_t2 - mse : 1.0840265793375672, mae : 0.8293172505332831
    #avg peformance (f_t1 & ft_2) - mse : 3.1494380533136925, mae : 1.3481589115142107

    # --------------------------------------------------------------------------
    # Tuned Model
    # --------------------------------------------------------------------------
    # Define parameter grid
    param_grid = {
        "rf__max_depth" : np.arange(10,120,10),
        "rf__n_estimators" : np.arange(100,1000, 100),
        "rf__min_samples_split" : stats.randint(1,10),
        "rf__min_samples_leaf" : stats.randint(1,5)
    }

    # Train Model
    
    search = train_model(param_grid, (X_train, y_train))

    print(f"Best Params : {search.best_params_}")
    print(f"Best train score : {search.best_score_}")

    # Cv results
    results, p = get_cv_result(search)
    results = results[["param_rf__max_depth","param_rf__n_estimators","param_rf__min_samples_split","param_rf__min_samples_leaf","mean_train_score","std_train_score"]]
    print(results.head())
    p.show()
    
    # --------------------------------------------------------------------------
    # Performance Evaluation on testset
    # --------------------------------------------------------------------------

    # Check performance Metrics
    performance_measures = eval_metrics(search, (X_test, y_test))

    
    with mlflow.start_run():
        mlflow.log_param("max_depth", search.best_params_["rf__max_depth"])
        mlflow.log_param("N_estimators", search.best_params_["rf__n_estimators"])
        mlflow.log_param("min_samples_split", search.best_params_["rf__min_samples_split"])
        mlflow.log_param("min_samples_leaf", search.best_params_["rf__min_samples_leaf"])
        mlflow.log_metric("avg_train_mse", search.best_score_)
        mlflow.log_metric("avg_test_mse", performance_measures["ft_avg_mse"])
        mlflow.log_metric("avg_test_mae", performance_measures["ft_avg_mae"])
        mlflow.log_metric("ft1_test_mse", performance_measures["ft1_mse"])
        mlflow.log_metric("ft1_test_mae", performance_measures["ft1_mae"])
        mlflow.log_metric("ft2_test_mse", performance_measures["ft2_mse"])
        mlflow.log_metric("ft2_test_mae", performance_measures["ft2_mae"])

        mlflow.sklearn.log_model(search, "models/knn_model")
    