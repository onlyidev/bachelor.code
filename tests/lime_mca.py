# %%
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import sklearn
import pandas as pd
import mlflow
import dvc.api
import sklearn.ensemble
#%%
params = dvc.api.params_show()
mca_data = pd.read_csv(params["train"]["mca"])
data_ben = np.load(f"{params['train']['benign']}")
data_mal = np.load(f"{params['train']['malware']}")
df_ben = pd.DataFrame(data_ben)
df_mal = pd.DataFrame(data_mal)
#%%
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
rf.fit(mca_data.iloc[:,:-1], mca_data.iloc[:,-1])
#%%
explainer = LimeTabularExplainer(mca_data.values[:,:-1], feature_names=mca_data.columns[:-1], class_names=['Benign', 'Malicious'])
detector = mlflow.sklearn.load_model(f"runs:/{params['malgan']['id']}/BB")
gen = mlflow.pyfunc.load_model(f"runs:/{params['malgan']['id']}/generator")
mca = mlflow.pyfunc.load_model(f"runs:/{params['malgan']['id']}/mca")

# %%
normal = set()

for i, obf in df_ben.iterrows():
    if i > 25:
        break
    exp = explainer.explain_instance(mca.predict([obf.values]).values[0], rf.predict_proba)
    normal = normal | set([name for name, _ in exp.as_list()])

#%%
obf = gen.predict(df_mal.head(1))
exp = explainer.explain_instance(mca.predict(obf).values[0], rf.predict_proba)
exp.show_in_notebook(show_table=True, show_all=False)
#%%
[a in normal for a in set([name for name, _ in exp.as_list()])]