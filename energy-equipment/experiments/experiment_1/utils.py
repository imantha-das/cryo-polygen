import pandas as pd 
import numpy as np 
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(path):

    # load data
    df = pd.read_excel(path)

    # rename columns
    df.rename(columns = {
        "x1" : "part_load",
        "x2" : "water_temp",
        "x3" : "gas_temp",
        "x4" : "ht_coef",
        "y1" : "f_thickness1",
        "y2" : "f_thickness2"
    }, inplace = True)

    # convert response variable to a continous value
    df = df.assign(
        f_thickness1 = df["f_thickness1"].astype("float"),
        f_thickness2 = df["f_thickness2"].astype("float")
    )

    return df

# ------------------------------------------------------------------------------
# Get CV results
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
# Evaluation
# ------------------------------------------------------------------------------
def eval_metrics(model, test_set:tuple):

    X_test, y_test = test_set
    yhat_test = model.predict(X_test)

    ft1_mse = mean_squared_error(y_test[:,0],yhat_test[:,0])
    ft1_mae = mean_absolute_error(y_test[:,0],yhat_test[:,0])
    ft2_mse = mean_squared_error(y_test[:,1],yhat_test[:,1])
    ft2_mae = mean_absolute_error(y_test[:,1],yhat_test[:,1])
    ft_avg_mse = mean_squared_error(y_test, yhat_test)
    ft_avg_mae = mean_absolute_error(y_test, yhat_test)

    performance_measures = {
        "ft1_mse" : ft1_mse,
        "ft1_mae" : ft1_mae,
        "ft2_mse" : ft2_mse,
        "ft2_mae" : ft2_mae,
        "ft_avg_mse" : ft_avg_mse,
        "ft_avg_mae" : ft_avg_mae
    }

    print("\n Performance on testing set : ")
    print(f"f_t1 - mse : {ft1_mse}, mae : {ft1_mae}")
    print(f"f_t2 - mse : {ft2_mse}, mae : {ft2_mae}")
    print(f"avg peformance (f_t1 & ft_2) - mse : {ft_avg_mse}, mae : {ft_avg_mae}")

    return performance_measures