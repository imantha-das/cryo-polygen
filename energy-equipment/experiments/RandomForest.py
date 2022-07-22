import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import train_test_split

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

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")

# Normalising - Not Required for RF

# Grid Search
param_grid = [
    {
        "max_depth" : [1,5,10,20,None],
        "n_estimators" : [32, 64, 100, 200],
        "min_sample_split" : [0.1,0.3,0.5,1]
        "min_sample_leaf"
    }
]
