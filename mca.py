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
data_benign = np.load(f"{params['benign']}")
df_benign = pd.DataFrame(data_benign)

data_malicious = np.load(f"{params['malware']}")
df_malicious = pd.DataFrame(data_malicious)

df = pd.concat([df_benign, df_malicious], ignore_index=True)

# Display the DataFrame
df.head()
#%%
mca = prince.MCA()
mca = mca.fit(df)
transformed_data = mca.transform(df)
transformed_data.columns = ["Low API Density", "High API Density"]

# Concatenate transformed_data with the original DataFrame df
df = pd.concat([df, transformed_data], axis=1)

# Separate the transformed data back into benign and malicious
benign = transformed_data.iloc[: len(df_benign)]
mal = transformed_data.iloc[len(df_benign) :]

# Create labels for each class
labels = [0] * len(df_benign) + [1] * len(df_malicious)

# Add the labels as a new column to the transformed data
df["class"] = labels

# %%
df.iloc[:,-3:].to_csv(f"{params['mca']}", index=False)

# %%skip
helpers.mcaflow.log_model(run_id, mca, "mca", input_example=df_benign.head(), registered_model_name="MCA")