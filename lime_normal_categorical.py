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
#%% Parameters
t_params, d_params = load_params("train", "detector")
#%% Load models
explainer = helpers.lime.CategoricalLimeExplainer()
classifier = mlflow.sklearn.load_model(f"runs:/{d_params['id']}/BB")
#%% Get set
df = pd.DataFrame(np.load(t_params["benign"], mmap_mode='r'))

normal = set()

def process_example(example):
    exp = explainer.explain_important(example, classifier)
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
with open(t_params["normal_categorical_features"], "w") as f:
    f.write(str(normal))
