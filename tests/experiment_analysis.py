#%%
from limeVerify import LimeVerify, HashableType
from helpers.params import load_params
import pandas as pd
import numpy as np
import mlflow

mca_params, mca_cls_params, t_params, m_params, v_params = load_params("mca", "mca_classifier", "train", "malgan", "valid")

#%%

verifier = LimeVerify(dict(
            mca=mca_params["id"], mca_cls=mca_cls_params["id"]), t_params["normal_features"], t_params["mca"], 18)
obfuscator = mlflow.pyfunc.load_model(f"runs:/{m_params['id']}/generator")
# %%

df = pd.DataFrame(np.load(v_params["benign"], mmap_mode='r')).head(5)
# df = obfuscator.predict(df)
t = verifier.transform(HashableType(df, "f"))

#%%
verifier.verify(HashableType(t.iloc[4].values, compareByKey=False), outputResult=True)