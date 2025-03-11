r"""
    Collect a normal set of features using LIME and MCA data
"""
#%% Imports
import numpy as np
import dvc.api
import pandas as pd
import mlflow
import helpers.lime
from tqdm import tqdm
import multiprocessing
import warnings

warnings.filterwarnings("ignore")
#%% Parameters
params = dvc.api.params_show()
mca_data = pd.read_csv(params["train"]["mca"])
#%% Load models
explainer = helpers.lime.LimeExplainer(params["train"]["mca"])
mca = mlflow.pyfunc.load_model(f"runs:/{params['experiment']['id']}/mca")
mca_classifier = mlflow.sklearn.load_model(f"runs:/{params['experiment']['id']}/mca_classifier")

#%% Transform dataset
print("Selecting benign MCA features")
df = mca_data.loc[mca_data["class"] == 0].iloc[:,:-1]
#%% Get set
normal = set()

def process_example(example):
    exp = explainer.explain_important(example, mca_classifier)
    return set([name for name, _ in exp.as_list()])

if __name__ == '__main__':
    print("Processing benign examples")
    with multiprocessing.Pool(processes=4) as pool:
        results = tqdm(pool.imap_unordered(process_example, [example for _, example in df.iterrows()]), total=len(df), desc="Collecting normal features")
        iter = tqdm(results, desc="Combining results")
        for result in iter:
            normal |= result
            iter.set_description(f"Combining results ({len(normal)} features)")
    
#%% Save set
with open(params["train"]["normal_features"], "w") as f:
    f.write(str(normal))
