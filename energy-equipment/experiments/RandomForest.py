import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import train_test_split, GridSearchCV

# ------------------------------------------------------------------------------
# Import Data
# ------------------------------------------------------------------------------

df = pd.read_excel("data/Dataset1_prediction.xlsx")
df.rename(columns = {
    "x1" : "part_load",
    "x2" : "water_temp",
    "x3" : "gas_temp",
    "x4" : "ht_coef",
    "y1" : "f_thickness1",
    "y2" : "f_thickness2"
}, inplace = True)

# Convert Response Variable to a continous value
df = df.assign(
    f_thickness1 = df["f_thickness1"].astype("float"),
    f_thickness2 = df["f_thickness2"].astype("float")
)


# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------
# Drop any missing values if any - None in this case 
df = df.dropna()

# Train-Test-Split
X = df.drop(["f_thickness1", "f_thickness2"],axis = 1)
y = df[["f_thickness1", "f_thickness2"]]

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
print(f"X_train : {X_train.shape}, {type(X_train)}")
print(f"y_train : {y_train.shape}, {type(X_test)}")
print(f"X_test : {X_test.shape}", {type(y_train)})
print(f"y_test : {y_test.shape}", type(y_test))

X_train = X_train.values 
X_test = X_test.values 
y_train = y_train.values 
y_test = y_test.values

# Normalising - Not Required for RF

# ------------------------------------------------------------------------------
# Define Hyperparameters
# ------------------------------------------------------------------------------
# Grid Search
param_grid = [
    {
        "bootstrap" : ["True", "False"],
        "max_depth" : [None,10,20,50,100],
        "min_samples_leaf" : [1,2,4],
        "min_samples_split" : [2,5,10],
        "n_estimators" : [100,200,500,1000]
    }
]

# ------------------------------------------------------------------------------
# Base - Model
# ------------------------------------------------------------------------------

"""
forest = RandomForestRegressor()
forest.fit(X_train, y_train)

preds = forest.predict(X_test)
rmse = lambda x1,x2,idx: np.sqrt(mean_squared_error(x1[:,idx],x2[:,idx]))

print(f"RMSE f1_t : {rmse(preds,y_test,0)}")
print(f"RMSE f2_t : {rmse(preds,y_test,1)}")
"""

# ------------------------------------------------------------------------------
# Grid Search
# ------------------------------------------------------------------------------

forest = RandomForestRegressor()
grid_search = GridSearchCV(
    estimator = forest,
    param_grid = param_grid,
    cv = 5,
    verbose = 3,
    scoring = ("mean_squared_error", "mean_absolute_error")
)
grid_search.fit(X_train, y_train)

print(f"Score : {grid_search.score(X_test, y_test)}")
print(f"Best model : {}")

