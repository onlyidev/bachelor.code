"""
    Train MCA classifier - Random Forest
    It's job is the same as main classifier, but it uses the MCA data
"""
#%%
import mlflow
import dvc.api
import sklearn.ensemble
import sklearn.model_selection
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
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=9)
    # rf = sklearn.ensemble.HistGradientBoostingClassifier(verbose=True)
    # param_grid = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'max_iter': [100, 200, 500],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_leaf': [20, 50, 100],
    #     'l2_regularization': [0.0, 1.0, 10.0],
    # }
    # search = sklearn.model_selection.RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    logger.info("Begin fit")
    rf.fit(mca_data.iloc[:,:-1], mca_data.iloc[:,-1])
    logger.info("End fit")
    mlflow.sklearn.log_model(rf, "mca_classifier", registered_model_name="MCA_classifier", input_example=mca_data.iloc[:,:-1].head(1))
    helpers.experiment.exportRunYaml(key="mca_classifier")