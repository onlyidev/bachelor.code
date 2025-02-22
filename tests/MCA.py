# %%
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt

# Load the .npy file
data_benign = np.load('../data/trial_ben.npy')
df_benign = pd.DataFrame(data_benign)

data_malicious = np.load('../data/trial_mal.npy')
df_malicious = pd.DataFrame(data_malicious)

df = pd.concat([df_benign, df_malicious], ignore_index=True)

# Display the DataFrame
df.head()

mca = prince.MCA()
mca = mca.fit(df)
transformed_data = mca.transform(df)

# Separate the transformed data back into benign and malicious
benign = transformed_data.iloc[:len(df_benign)]
mal = transformed_data.iloc[len(df_benign):]

# Create labels for each class
labels = ['benign'] * len(df_benign) + ['malicious'] * len(df_malicious)

# Add the labels as a new column to the transformed data
transformed_data['class'] = labels

# %%
plt.scatter(benign.iloc[:, 0], benign.iloc[:, 1], label='Benign')
plt.scatter(mal.iloc[:, 0], mal.iloc[:, 1], label='Malicious')
plt.legend()
plt.show()

# %%
transformed_data.to_csv('../data/MCA.csv', index=False)
