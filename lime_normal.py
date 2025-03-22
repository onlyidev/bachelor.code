r"""
    Collect a normal set of features using LIME and MCA data
"""
#%% Imports
import numpy as np
import pandas as pd
import mlflow
import helpers.lime
from tqdm import tqdm
import warnings
from helpers.params import load_params

warnings.filterwarnings("ignore")
tqdm.pandas()
#%% Parameters
t_params, mca_params, mca_cls_params = load_params("train", "mca", "mca_classifier")
mca_data = pd.read_csv(t_params["mca"])
#%% Load models
explainer = helpers.lime.LimeExplainer(t_params["mca"])
mca = mlflow.pyfunc.load_model(f"runs:/{mca_params['id']}/mca")
mca_classifier = mlflow.sklearn.load_model(f"runs:/{mca_cls_params['id']}/mca_classifier")

#%% Transform dataset
print("Selecting benign MCA features")
df = mca_data.loc[mca_data["class"] == 0].iloc[:,:-1]
#%% Get set
def extract(example):
    exp = explainer.explain_all(example, mca_classifier)
    return [(k, -v if v < 0 else None) for k, v in exp.as_list()]


if __name__ == '__main__':
    print("Processing benign examples")
    results = df.progress_apply(extract, axis=1) # type: ignore
    rdf = pd.DataFrame(np.stack(results.to_list()).reshape([-1,2]))
    rdf.columns = ["feature", "importance"]
    rdf["importance"] = pd.to_numeric(rdf["importance"], errors='coerce')
    rdf = rdf.dropna()
    rdf = rdf.groupby("feature")["importance"].sum().sort_values(ascending=False) # type: ignore
    
    num_features = int(np.log(len(df)))
    print(f"Selecting {num_features} most important features")
    #%% Save set
    with open(t_params["normal_features"], "w") as f:
        f.write(str(set(rdf.head(num_features).keys().to_list())))
