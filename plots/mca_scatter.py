#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#%%
df = pd.read_csv('../data/MCA.csv')
# %%
df = df.iloc[:, [0,1,-1]]
dfmal = df.where(df.iloc[:, -1] == 1).dropna()
dfben = df.where(df.iloc[:, -1] == 0).dropna()
#%%
plt.ylim(-1,1)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.scatter(dfben.iloc[:, 0], dfben.iloc[:, 1], s=1, color="blue", label="Nekenkėjiška")
plt.scatter(dfmal.iloc[:, 0], dfmal.iloc[:, 1], s=1, color="orange", label="Kenkėjiška", marker='x')
plt.legend()
plt.savefig('../plots/mca_scatter.png', dpi=300, bbox_inches='tight')
#%%
