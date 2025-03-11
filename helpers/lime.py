from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np

class LimeExplainer:
    C = 100000 # LIME scaling constant
    def __init__(self, data):
        self.explainer = self.__getExplainer(data)
    
    def explain(self, example, classifier, **kwargs):
        exp = self.explainer.explain_instance(example*self.C, lambda x: classifier.predict_proba(x/self.C), **kwargs)
        assert exp.available_labels() == [1]
        return exp

    def explain_important(self, example, classifier):
        return self.explain(example, classifier, num_features=int(np.sqrt(self.num_features)))

    def __getExplainer(self, data):
        mca_data = pd.read_csv(data) * self.C # Scale data for LIME
        self.num_features = len(mca_data.columns) - 1
        return LimeTabularExplainer(mca_data.values[:,:-1], feature_names=mca_data.columns[:-1], class_names=['Benign', 'Malicious'], discretize_continuous=True, sample_around_instance=True)
