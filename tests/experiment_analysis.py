#%%
from limeVerify import LimeVerify, HashableType
from helpers.params import load_params
import pandas as pd
import numpy as np

mca_params, mca_cls_params, t_params, m_params, v_params = load_params("mca", "mca_classifier", "train", "malgan", "valid")

#%%

verifier = LimeVerify()
# %%

df = pd.DataFrame(np.load(v_params["benign"], mmap_mode='r')).head(5)
# df = obfuscator.predict(df)
t = verifier.transform(HashableType(df, "f"))

#%%
verifier.verify(HashableType(t.iloc[4].values, compareByKey=False), outputResult=True)