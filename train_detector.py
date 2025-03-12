import sklearn.ensemble
import pandas as pd
import numpy as np
import logging
from helpers.params import load_params
from helpers.experiment import *

logger = logging.getLogger()
s_params, t_params, e_params = load_params("split", "train", "experiment")

with startExperiment(run_id=e_params["id"]) as exp:
    logs()
    detector = sklearn.ensemble.RandomForestClassifier(n_estimators=t_params["estimators"])
    df_ben = pd.DataFrame(np.load(t_params["benign"]))
    df_mal = pd.DataFrame(np.load(t_params["malware"]))
    df_ben["class"] = 0
    df_mal["class"] = 1
    df = pd.concat([df_ben, df_mal], ignore_index=True)
    df = df.sample(frac=1, random_state=s_params["random_state"]).reset_index(drop=True)
    
    logger.info("Fitting the detector")
    detector.fit(df.drop(columns=['class']), df['class'])
    logger.info("Detector fitted")
    
    mlflow.sklearn.log_model(detector, "BB", registered_model_name="Malware detector", input_example=df.drop(columns=['class']).head(1))
