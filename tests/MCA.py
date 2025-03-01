# %%
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt
import dvc.api

params = dvc.api.params_show()["train"]
# Load the .npy file
data_benign = np.load(f"data/{params['benign']}")
df_benign = pd.DataFrame(data_benign)

data_malicious = np.load(f"data/{params['malware']}")
df_malicious = pd.DataFrame(data_malicious)

df = pd.concat([df_benign, df_malicious], ignore_index=True)

# Display the DataFrame
df.head()

mca = prince.MCA()
mca = mca.fit(df)
transformed_data = mca.transform(df)
transformed_data.columns = ['Low API Density', 'High API Density']

# Concatenate transformed_data with the original DataFrame df
df = pd.concat([df, transformed_data], axis=1)

# Separate the transformed data back into benign and malicious
benign = transformed_data.iloc[:len(df_benign)]
mal = transformed_data.iloc[len(df_benign):]

# Create labels for each class
labels = [0] * len(df_benign) + [1] * len(df_malicious)

# Add the labels as a new column to the transformed data
df['class'] = labels

# %%
plt.scatter(benign.iloc[:, 0], benign.iloc[:, 1], label='Benign')
plt.scatter(mal.iloc[:, 0], mal.iloc[:, 1], label='Malicious')
plt.legend()
plt.show()

# %%
df.to_csv(f"data/{params['mca']}", index=False)
