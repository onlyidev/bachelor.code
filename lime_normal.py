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
import multiprocessing
import warnings

warnings.filterwarnings("ignore")
#%% Parameters
params = dvc.api.params_show()
mca_data = pd.read_csv(params["train"]["mca"])
data_ben = np.load(f"{params['train']['benign']}")
df_ben = pd.DataFrame(data_ben)
#%% Load models
explainer = LimeTabularExplainer(mca_data.values[:,:-1], feature_names=mca_data.columns[:-1], class_names=['Benign', 'Malicious'])
mca = mlflow.pyfunc.load_model(f"runs:/{params['experiment']['id']}/mca")
mca_classifier = mlflow.sklearn.load_model(f"runs:/{params['experiment']['id']}/mca_classifier")

#%% Transform dataset
print("Transforming benign dataset to MCA values")
df = mca.predict(df_ben)
#%% Get set
normal = set()

def process_example(example):
    exp = explainer.explain_instance(example, mca_classifier.predict_proba, num_samples=2000)
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
