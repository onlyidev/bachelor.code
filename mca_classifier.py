"""
    Train MCA classifier - Random Forest
    It's job is the same as main classifier, but it uses the MCA data
"""
#%%
import mlflow
import dvc.api
import sklearn.ensemble
import pandas as pd
import helpers.experiment
import logging

logger = logging.getLogger()

#%%
params = dvc.api.params_show()
t_params = params["train"]
mca_data = pd.read_csv(params["train"]["mca"])

#%%
with helpers.experiment.startExperiment(name=t_params["name"], run_name="mca_classifier") as exp:
    helpers.experiment.logs()
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=t_params["estimators"])
    logger.info("Begin fit")
    rf.fit(mca_data.iloc[:,:-1], mca_data.iloc[:,-1])
    logger.info("End fit")
    mlflow.sklearn.log_model(rf, "mca_classifier", registered_model_name="MCA_classifier", input_example=mca_data.iloc[:,:-1].head(1))
    helpers.experiment.exportRunYaml(key="mca_classifier")