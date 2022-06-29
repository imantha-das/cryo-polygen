# Energy Forecasting

## Description

The folder contains models to forecast the energy load in the next 1 or 3 hours.

## Files & Folders

- data : Contains energy load data in `.csv` format
- model : All files relevent to predictive models
  - state_dict : Trained weights for a particular model
  - hyperparams.json :  Best tuned hyperparameters
  - models.py : Machine learning models
- forecast.py : Python script to forecast the energy load in the next 1 or 3 hours
- seach.py + trial.py : Used for hyperparameter tuning 
- train.py : Used to train the model
- misc : Contains extra models, data explorations, images etc 

## General Usage

### Hyperparameter tuning

- Run search.py with the relevent hyperparameters indicated in `search_space` varaible (line 9)
  - `python search.py`
- search.py will call trial.py which will do repetative training on the specified parameter combinations.
- The best hyperparameter combination will be saved in hyperparams.json at the end of training.

### Training model

- Run train.py which will train the model (weights + biases) on the best hyperparameter combinations.
  - `python train.py`
- Model weights and bias will be saved in state_dict

### Forecast
- Run forecast.py to forecast the energy load in the next 1 or 3 hours. Use 1 or 3 as an optional argument to the script (default 1).
  - i.e `python forecast.py --hours 3`