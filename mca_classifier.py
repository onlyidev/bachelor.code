#%%
import mlflow
import dvc.api
import sklearn.ensemble
import pandas as pd
import helpers.experiment

#%%
params = dvc.api.params_show()
t_params = params["train"]
e_params = params["experiment"]
mca_data = pd.read_csv(params["train"]["mca"])

#%%
with helpers.experiment.startExperiment(run_id=params["experiment"]["id"]):
    helpers.experiment.logs()
    mlflow.log_params(t_params)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=t_params["estimators"])
    rf.fit(mca_data.iloc[:,:-1], mca_data.iloc[:,-1])
    mlflow.sklearn.log_model(rf, "mca_classifier", registered_model_name="MCA_classifier", input_example=mca_data.iloc[:,:-1].head(1))