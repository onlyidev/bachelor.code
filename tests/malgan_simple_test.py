# %%
import mlflow
import pandas as pd
import numpy as np
import sklearn

# %%
ben = np.load('data/raw/trial_ben.npy')
mal = np.load('data/raw/trial_mal.npy')
print(ben,mal)

# %%

id = 'runs:/1554c8a5c3eb4c28a0ef268c56b39432'

black_box = mlflow.pyfunc.load_model(f"{id}/BB")
gen = mlflow.pyfunc.load_model(f"{id}/generator")
print(black_box, gen)


# %%
entry = mal[0]
initial_pred = black_box.predict(pd.DataFrame(mal))
obf = gen.predict(pd.DataFrame(mal))

# %%
mod_pred = black_box.predict(obf)
sklearn.metrics.f1_score(np.ones(mod_pred.shape), mod_pred, average='binary') # F1 score