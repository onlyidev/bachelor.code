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
id = 'runs:/1554c8a5c3eb4c28a0ef268c56b39432'
gen = mlflow.pyfunc.load_model(f"{id}/generator")
obf = gen.predict(pd.DataFrame(mca_data.iloc[443,:-3].values.reshape(1,-1)))
#%%
# exp = explainer.explain_instance(mca_data.iloc[443,:-3].values, rf.predict_proba, num_features=100)
exp = explainer.explain_instance(obf.iloc[0].values, rf.predict_proba, num_features=100)
# %%
exp.show_in_notebook(show_table=True, show_all=False)

#%%
mca_data.iloc[443]["class"]