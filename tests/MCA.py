# %%
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('../data/trial_ben.npy')

# Convert the numpy array to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
df.head()

mca = prince.MCA()
mca = mca.fit(df)
benign = mca.transform(df)

# %%
# Load the .npy file
data = np.load('./data/trial_mal.npy')

# Convert the numpy array to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
df.head()

mca = prince.MCA()
mca = mca.fit(df)
mal = mca.transform(df)

# %%
plt.scatter(benign.iloc[:, 0], benign.iloc[:, 1], label='Benign')
plt.scatter(mal.iloc[:, 0], mal.iloc[:, 1], label='Malicious')
plt.legend()
plt.show()

# %%
benign.to_csv()


