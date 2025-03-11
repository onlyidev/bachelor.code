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
        confusion = confusion_matrix(self.y, y_pred)
        print(report)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['Benign', 'Malware'])
        disp.plot().figure_.savefig(m_params["normal_confusion"])
        with open(m_params["normal"], "w") as f:
            f.write(self.metrics(y_pred))
        
        
if __name__ == '__main__':
    e_params, m_params, *_, v_params = load_params()
    args = sys.argv[1:]
    if len(args) < 1:
        raise ValueError("Please provide the experiment type")
    if args[0] == "normal":
        logger.info("Running normal case")
        NormalCase().run()
    else:
        raise ValueError("Invalid experiment type")
