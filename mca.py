# %%
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt
import helpers.mcaflow
from helpers.experiment import *
from helpers.params import load_params
import logging

logger = logging.getLogger()

t_params, = load_params("train")
# Load the .npy file
# Issue URL: https://github.com/onlyidev/bachelor.code/issues/5
#
data_benign = np.load(f"{t_params['benign']}", mmap_mode='r')
df_benign = pd.DataFrame(data_benign)

data_malicious = np.load(f"{t_params['malware']}", mmap_mode='r')
df_malicious = pd.DataFrame(data_malicious)

df = pd.concat([df_benign, df_malicious], ignore_index=True)

#%%
# TODO Add inertia calculations / graph
# Issue URL: https://github.com/onlyidev/bachelor.code/issues/4
#
with startExperiment(t_params["name"], run_name="mca") as exp:
    mca = prince.MCA(n_components=t_params["mca_components"], n_iter=3)
    mca = mca.fit(df)
    inertia = max(mca.cumulative_percentage_of_variance_)
    logger.info(f"Explained inertia: {inertia}",)
    assert inertia > 50, "Inertia is too low, increase the number of components" 
    
    transformed_data = mca.transform(df)
    table = {
        inertia: mca.cumulative_percentage_of_variance_,
    }

    # Create labels for each class
    labels = [0] * len(df_benign) + [1] * len(df_malicious)
    transformed_data["class"] = labels

    # %%
    transformed_data.to_csv(f"{t_params['mca']}", index=False)

    # %%skip
    helpers.mcaflow.log_model(mca, "mca", input_example=df_benign.head(), registered_model_name="MCA")
    mlflow.log_table(data=table, artifact_file="inertia.json")
    exportRunYaml(key="mca")