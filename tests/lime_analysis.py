from helpers.params import load_params
import mlflow
from helpers.lime import *
from tqdm import tqdm

t_params, mca_params, mca_cls_params, d_params = load_params("train", "mca", "mca_classifier", "detector")
tqdm.pandas()

#%%
e =  CategoricalLimeExplainer()
det = mlflow.sklearn.load_model(f"runs:/{d_params['id']}/BB")
#%%
df = pd.DataFrame(np.load(t_params["benign"], mmap_mode='r'))

#%%
def extract(sample):
    exp = e.explain(sample, det, num_features=200)
    return [(k, -v if v < 0 else None) for k, v in exp.as_list()]

#%%
results = df.progress_apply(extract, axis=1) # type: ignore
rdf = pd.DataFrame(np.stack(results.to_list()).reshape([-1,2]))
rdf.columns = ["feature", "importance"]
rdf = rdf.dropna()
#%%
rdf.groupby("feature")["importance"].std().sort_values(ascending=False) # type: ignore