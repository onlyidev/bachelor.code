# %%
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt
import dvc.api
import helpers.mcaflow

params = dvc.api.params_show()
run_id = params["experiment"]["id"]
params = params["train"]
# Load the .npy file
data_benign = np.load(f"{params['benign']}", mmap_mode='r')
df_benign = pd.DataFrame(data_benign).head(100)

data_malicious = np.load(f"{params['malware']}", mmap_mode='r')
df_malicious = pd.DataFrame(data_malicious).head(100)

df = pd.concat([df_benign, df_malicious], ignore_index=True)

#%%

mca = prince.MCA(n_components=10, n_iter=3)
mca = mca.fit(df)
transformed_data = mca.transform(df)
# transformed_data.columns = ["Low API Density", "High API Density"]

# Create labels for each class
labels = [0] * len(df_benign) + [1] * len(df_malicious)
transformed_data["class"] = labels

# %%
transformed_data.to_csv(f"{params['mca']}", index=False)

# %%skip
helpers.mcaflow.log_model(run_id, mca, "mca", input_example=df_benign.head(), registered_model_name="MCA")
