r"""
    Collect a normal set of features using LIME and MCA data
"""
#%% Imports
import numpy as np
import pandas as pd
import mlflow
import helpers.lime
from tqdm import tqdm
import multiprocessing
import warnings
from helpers.params import load_params

warnings.filterwarnings("ignore")
tqdm.pandas()

#%% Parameters
t_params, d_params = load_params("train", "detector")
#%% Load models
explainer = helpers.lime.CategoricalLimeExplainer()
classifier = mlflow.sklearn.load_model(f"runs:/{d_params['id']}/BB")
#%% Get set
df = pd.DataFrame(np.load(t_params["benign"], mmap_mode='r'))

def extract(example):
    exp = explainer.explain_all(example, classifier)
    return [(k, -v if v < 0 else None) for k, v in exp.as_list()]

if __name__ == '__main__':
    print("Processing benign examples")
    results = df.progress_apply(extract, axis=1) # type: ignore
    rdf = pd.DataFrame(np.stack(results.to_list()).reshape([-1,2]))
    rdf.columns = ["feature", "importance"]
    a = rdf.groupby("feature")["importance"].mean().sort_values(ascending=False) # type: ignore
    s = rdf.groupby("feature")["importance"].std().sort_values(ascending=False) # type: ignore
    eTable = a.rename("average").to_frame().join(s.rename("std")).dropna()

    #%% Save set
    with open(t_params["normal_categorical_features"], "w") as f:
        eTable.to_csv(f)
