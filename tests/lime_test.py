# %%
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import sklearn
import pandas as pd
import mlflow
#%%
mca_data = pd.read_csv('data/MCA.csv')

#%%
explainer = LimeTabularExplainer(mca_data.values[:,:-3], feature_names=mca_data.columns[:-3], class_names=['Benign', 'Malicious'], categorical_features=[*range(0,mca_data.shape[1]-3)])

#%%
ben = mca_data[mca_data['class'] == 0].iloc[:,:-3].values
mal = mca_data[mca_data['class'] == 1].iloc[:,:-3].values 
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
rf.fit(np.concatenate((ben, mal)), np.concatenate((np.zeros(ben.shape[0]), np.ones(mal.shape[0]))))
# %%
exp = explainer.explain_instance(mca_data.iloc[900,:-3].values, rf.predict_proba, num_features=100)
# %%
exp.show_in_notebook(show_table=True, show_all=False)

#%%
mca_data.iloc[900]["class"]