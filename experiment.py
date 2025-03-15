r"""
    This is the experiment with and w/o the use of the proposed method.
"""

import sys
import numpy as np
import pandas as pd
from helpers.params import load_params
import mlflow
from helpers.logging import *
import logging
import abc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import json
import limeVerify

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

class Experiment:
    @timing
    def __init__(self):
        self.__loadData()
        self.detector = mlflow.sklearn.load_model(f"runs:/{e_params['id']}/BB")
    
    @log
    def __loadData(self, random_state=42):
        df_ben = pd.DataFrame(np.load(v_params["benign"]))
        df_mal = pd.DataFrame(np.load(v_params["malware"]))
        df_obf = self.__obfuscateData(df_mal)
        df_ben['class'] = 0  # Benign = 0
        df_mal['class'] = 1  # Malware = 1
        df_obf['class'] = 1
        
        self.df = pd.concat([df_ben, df_mal, df_obf], ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    @log
    def __obfuscateData(self, data):
        gen = mlflow.pyfunc.load_model(f"runs:/{e_params['id']}/generator")
        return gen.predict(data)
    
    @property
    def X(self):
        return self.df.drop(columns=['class'])
    
    @property
    def y(self):
        return self.df['class']  
    
    @abc.abstractmethod
    def run(self):
        pass
    
    def metrics(self, y_pred):
        y_true = self.y
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        return json.dumps(metrics, indent=2)


class NormalCase(Experiment):
    @log
    @timing
    def run(self):
        y_pred = self.detector.predict(self.X)
        report = classification_report(self.y, y_pred)
        confusion = confusion_matrix(self.y, y_pred, normalize='true')
        print(report)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['Benign', 'Malware'])
        disp.plot().figure_.savefig(m_params["normal_confusion"])
        with open(m_params["normal"], "w") as f:
            f.write(self.metrics(y_pred))
            
class LimeCase(Experiment):
    
    @timing
    def __init__(self):
        super().__init__()
        self.verifier = limeVerify.LimeVerify(e_params["id"], t_params["normal_features"], t_params["mca"])
    
    @log
    @timing
    def verify(self, preds):
        pdf = pd.DataFrame(preds)
        df = pdf[pdf[0] == 0]
        features = self.X[self.X.index.isin(df.index)]
        t = self.verifier.transform(features)
        v = t.apply(lambda x: self.verifier.verify(x.values), axis=1)
        vt = v.transform(lambda x: 0 if x else 1)
        pdf.update(vt, overwrite=True)
        return pdf.values
    
    @log
    @timing
    def run(self):
        y_pred = self.detector.predict(self.X) # Original prediction
        y_pred = self.verify(y_pred)
        report = classification_report(self.y, y_pred)
        confusion = confusion_matrix(self.y, y_pred, normalize='true')
        print(report)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['Benign', 'Malware'])
        disp.plot().figure_.savefig(m_params["lime_confusion"])
        with open(m_params["lime"], "w") as f:
            f.write(self.metrics(y_pred))
        
        
if __name__ == '__main__':
    e_params, m_params, t_params, v_params = load_params("experiment", "metrics", "train", "valid")
    args = sys.argv[1:]
    if len(args) < 1:
        raise ValueError("Please provide the experiment type")
    if args[0] == "normal":
        logger.info("Running normal case")
        NormalCase().run()
    if args[0] == "lime":
        logger.info("Running LIME case")
        LimeCase().run()
    else:
        raise ValueError("Invalid experiment type")
