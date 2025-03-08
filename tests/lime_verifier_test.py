#%%
import limeVerify
import numpy as np
import pandas as pd
import mlflow
import dvc.api

params = dvc.api.params_show()
t_params = params["train"]
v_params = params["valid"]

#%%
df = pd.DataFrame(np.load(v_params["benign"])).head(5)
#%%
v = limeVerify.LimeVerify(params["experiment"]["id"], t_params["normal_features"], t_params["mca"])

#%%
t = v.transform(df)

#%%
exp = v.verify(t.iloc[2].values, outputResult=True)

#%%
gen = mlflow.pyfunc.load_model(f"runs:/{params['experiment']['id']}/generator")

#%%
obf = gen.predict(df)