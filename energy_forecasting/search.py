from nni.experiment import Experiment
from pathlib import Path
import time
import json
import os
from termcolor import colored

# Hyperparam search space
search_space = {
    "model" : {"_type" : "choice", "_value" : ["lstm"]},
    "hidden_size" : {"_type" : "choice", "_value" : [32,64]},
    "num_layers" : {"_type" : "choice", "_value" : [1,2,3]},
    "optimizer" : {"_type" : "choice", "_value" : ["adam","sgd","adamax"]},
    "learning_rate" : {"_type" : "choice", "_value" : [0.001, 0.005, 0.01, 0.05, 0.1]},
    "window_size" : {"_type" : "choice", "_value" : [24,72,168]},
    "batch_size" : {"_type" : "choice", "_value" : [1,32,64,128,256]},
    "epochs" : {"_type" : "choice", "_value" : [50,100,500]},
}

# Maximum number of trials 
max_trials = 30

# ------------------------------------------------------------------------------
# Search Configuration
# ------------------------------------------------------------------------------

search = Experiment("local")

# search name
search.config.experiment_name = "energy forecating hyperparam tuning"

search.config.trial_concurrency = 2 #evaluates 2 hyperparams at a time (i think)
search.config.max_trial_number = max_trials # evaluates 30 (look above) sets of hyperparams
search.config.search_space = search_space #hyperparam search space 
search.config.trial_command = "python trial.py"
search.config.trial_code_directory = Path(__file__).parent #path to train.py

# Tuner settings 
search.config.tuner.name = "Evolution" #Randomly initializes a population based on search space, for each generation it chooses better one and do some mutation on them to get next generation
search.config.tuner.class_args["optimize_mode"] = "minimize" # tuner attempts to minimize the given metrics
search.config.tuner.class_args["population_size"] = 8 #evolution based metric, popultation_size > concurrency, the greater the size the better

search.start(8080)
executed_trails = 0
while True:
    trials = search.export_data() # Returns exported information
    if executed_trails != len(trials):
        executed_trails = len(trials)
        print(f"\nTrials : {executed_trails} / {max_trials}")
    if search.get_status() == "DONE":

        best_trial = min(trials, key = lambda t: t.value)
        print(f"Best trial params: {best_trial.parameter}, Best loss : {best_trial.value}")
        
        with open("model/hyperparams.json", "w") as f:
            json.dump(best_trial.parameter, f)

        input("Experiment is finished. Press any key to exit ...")
        break
    print(".", end = "")
    time.sleep(10)