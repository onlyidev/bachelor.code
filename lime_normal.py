r"""
    Collect a normal set of features using LIME and MCA data
"""
#%% Imports
import numpy as np
import dvc.api
import pandas as pd
import mlflow
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

#%% Parameters
params = dvc.api.params_show()
mca_data = pd.read_csv(params["train"]["mca"])
data_ben = np.load(f"{params['train']['benign']}")
df_ben = pd.DataFrame(data_ben)
#%% Load models
explainer = LimeTabularExplainer(mca_data.values[:,:-1], feature_names=mca_data.columns[:-1], class_names=['Benign', 'Malicious'])
mca = mlflow.pyfunc.load_model(f"runs:/{params['malgan']['id']}/mca")
mca_classifier = mlflow.pyfunc.load_model(f"runs:/{params['malgan']['id']}/mca_classifier")

#%% Get set
normal = set()
for i, obf in tqdm(df_ben.iterrows(), total=len(df_ben)):
    exp = explainer.explain_instance(mca.predict([obf.values]).values[0], mca_classifier.predict_proba)
    normal = normal | set([name for name, _ in exp.as_list()])
    
#%% Save set
with open(params["normal_features"], "w") as f:
    f.write(str(normal))