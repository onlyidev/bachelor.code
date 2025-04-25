# %%
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt
from helpers.experiment import *
from helpers.params import load_params
from matplotlib.ticker import FixedLocator, FixedFormatter

#%%
t_params, = load_params("train")

data_benign = np.load(f"{t_params['benign']}", mmap_mode='r')
df_benign = pd.DataFrame(data_benign)

data_malicious = np.load(f"{t_params['malware']}", mmap_mode='r')
df_malicious = pd.DataFrame(data_malicious)

df = pd.concat([df_benign, df_malicious], ignore_index=True)
mca = prince.MCA(n_components=t_params["mca_components"], n_iter=3)
mca = mca.fit(df)
# %%
fig, ax1 = plt.subplots()
ax1.tick_params(axis='y', colors="blue")
ax1.yaxis.set_label_text("Inercija, %  ")
ax1.xaxis.set_label_text("Komponentė")
ax1.yaxis.set_label_coords(0,1.02)
ax1.xaxis.set_minor_locator(FixedLocator([5]))
ax1.xaxis.set_minor_formatter(FixedFormatter(['5']))
ax1.yaxis.label.set_rotation(0)
x = np.arange(1, len(mca.percentage_of_variance_) + 1)
ax1.plot(x, mca.percentage_of_variance_, label="Komponentės inercija")
ax1.axvline(x=5, color="black", linestyle=":", label="Alkūnės taško ašis")

ax2 = ax1.twinx()
ax2.tick_params(axis='y', colors="orange")
ax2.yaxis.set_label_text("Sukaupta inercija, %")
ax2.yaxis.set_label_coords(0.9,1.05)
ax2.yaxis.label.set_rotation(0)
ax2.plot(mca.cumulative_percentage_of_variance_, label="Sukaupta inercija", color='orange', linestyle='--')
fig.tight_layout()
fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
plt.savefig("./plots/scree.png", dpi=300)