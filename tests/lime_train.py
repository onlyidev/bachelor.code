#%%
import helpers.limeflow
import dvc.api
import pandas as pd
import numpy as np
import mlflow
from lime.lime_tabular import LimeTabularExplainer

#%% Parameters
params = dvc.api.params_show()
mca_data = pd.read_csv(params["train"]["mca"])
data_ben = np.load(f"{params['train']['benign']}")
df_ben = pd.DataFrame(data_ben)
#%% Load models
explainer = LimeTabularExplainer(mca_data.values[:,:-1], feature_names=mca_data.columns[:-1], class_names=['Benign', 'Malicious'])

#%% Transform dataset
print("Transforming benign dataset to MCA values")
mca = mlflow.pyfunc.load_model(f"runs:/{params['experiment']['id']}/mca")
df = mca.predict(df_ben)

#%% Save
helpers.limeflow.log_model(params["experiment"]["id"], explainer, "lime", input_example=df.head(1), registered_model_name="LIME")
