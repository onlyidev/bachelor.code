import sklearn.ensemble
import pandas as pd
import numpy as np
import logging

import sklearn.model_selection
import sklearn.neural_network
from helpers.params import load_params
from helpers.experiment import *

logger = logging.getLogger()
s_params, t_params = load_params("split", "train")

with startExperiment(name=t_params["name"], run_name="detector") as exp:
    logs()
    # detector = sklearn.ensemble.RandomForestClassifier(n_estimators=t_params["estimators"], criterion="log_loss")
    # detector = sklearn.([256, 512, 512, 256], learning_rate="adaptive", activation="relu", verbose=True)
    param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0.0, 1.0, 10.0],
    }

    detector = sklearn.ensemble.HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.1)
    search = sklearn.model_selection.RandomizedSearchCV(detector, param_grid, n_iter=20, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    df_ben = pd.DataFrame(np.load(t_params["benign"]))
    df_mal = pd.DataFrame(np.load(t_params["malware"]))
    df_ben["class"] = 0
    df_mal["class"] = 1
    df = pd.concat([df_ben, df_mal], ignore_index=True)
    df = df.sample(frac=1, random_state=s_params["random_state"]).reset_index(drop=True)
    
    logger.info("Fitting the detector")
    search.fit(df.drop(columns=['class']), df['class'])
    logger.info("Detector fitted")
    
    mlflow.sklearn.log_model(search, "BB", registered_model_name="Malware detector", input_example=df.drop(columns=['class']).head(1))
    exportRunYaml(key="detector")