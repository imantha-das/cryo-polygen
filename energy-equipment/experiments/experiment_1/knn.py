import os
os.chdir("experiments/experiment_1")

from termcolor import colored

from utils import load_data, get_cv_result, eval_metrics

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

import mlflow 
import mlflow.sklearn

def train_model(param_grid:dict, train_set:tuple):
    
    # Get X_train, y_train
    X_train, y_train = train_set

    # Construct pipeline
    pipe = Pipeline([
        ("scaler",MinMaxScaler()),
        ("knn",KNeighborsRegressor())
    ])

    # Randomized Search
    search = GridSearchCV(
        pipe,
        param_grid,
        scoring = ("neg_mean_squared_error"),
        cv = KFold(n_splits = 5, shuffle = True),
        n_jobs = -1,
        refit = True,
        #n_iter = 60,
        verbose = 2,
        return_train_score = True
    )
    
    # Fit model
    search.fit(X_train, y_train)

    return search

if __name__ == "__main__":

    # Start MLFlow
    mlflow.set_experiment("knn (fouling thickness)")

    # Get Data
    df = load_data("../../data/Dataset1_prediction.xlsx")
    print(df.head())

    X = df.drop(["f_thickness1", "f_thickness2"], axis = 1).values 
    y = df[["f_thickness1", "f_thickness2"]].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    #print(f"\nX_train : {X_train.shape}, y_train : {y_train.shape}")
    #print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    # --------------------------------------------------------------------------
    # Default Model
    # --------------------------------------------------------------------------

    #knn = KNeighborsRegressor()
    #knn.fit(X_train, y_train)
    #print(knn.predict(X_test))

    # Base Model Performance Metrics
    #f_t1 - mse : 4.441521739130435, mae : 1.7347826086956522
    #f_t2 - mse : 1.4284782608695654, mae : 0.9565217391304348
    #avg peformance (f_t1 & ft_2) - mse : 2.935000000000001, mae : 1.345652173913043

    # --------------------------------------------------------------------------
    # Tune Model
    # --------------------------------------------------------------------------

    # Define parameter grid
    param_grid = {
        "knn__n_neighbors" : [3,5,7,9,11,13,15],
        "knn__weights" : ["uniform", "distance"],
        "knn__metric" : ["minkowski","euclidean","manhattan"]
    }

    # Train Model
    search = train_model(param_grid, (X_train, y_train))

    print(f"Best Params : {search.best_params_}")
    print(f"Best train score : {search.best_score_}")

    # --------------------------------------------------------------------------
    # Get CV Results
    # --------------------------------------------------------------------------

    results, p = get_cv_result(search)
    results = results[["param_knn__n_neighbors","param_knn__weights","param_knn__metric","mean_train_score","std_train_score"]]
    print(results.head())
    p.show()

    # --------------------------------------------------------------------------
    # Performance Evaluation on testset
    # --------------------------------------------------------------------------

    # Check performance Metrics
    performance_measures = eval_metrics(search, (X_test, y_test))

    print(search.best_params_["knn__n_neighbors"])

    
    with mlflow.start_run():
        mlflow.log_param("n_neighbors", search.best_params_["knn__n_neighbors"])
        mlflow.log_param("weights", search.best_params_["knn__weights"])
        mlflow.log_param("metric", search.best_params_["knn__metric"])
        mlflow.log_metric("avg_train_mse", search.best_score_)
        mlflow.log_metric("avg_test_mse", performance_measures["ft_avg_mse"])
        mlflow.log_metric("avg_test_mae", performance_measures["ft_avg_mae"])
        mlflow.log_metric("ft1_test_mse", performance_measures["ft1_mse"])
        mlflow.log_metric("ft1_test_mae", performance_measures["ft1_mae"])
        mlflow.log_metric("ft2_test_mse", performance_measures["ft2_mse"])
        mlflow.log_metric("ft2_test_mae", performance_measures["ft2_mae"])

        mlflow.sklearn.log_model(search, "models/knn_model")
    