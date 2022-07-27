import pandas as pd 
import numpy as np 
import plotly.express as px

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

    results_concise = results[[
        "param_svm__estimator__C",
        "param_svm__estimator__gamma",
        "param_svm__estimator__kernel",
        "mean_test_score",
        "std_test_score"
    ]]
    return results_concise, p