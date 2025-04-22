#%%
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
import warnings
from tqdm import tqdm
from scripts.notify import Notifier
#%%
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%%
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger()
tqdm.pandas(desc="Verifying")

#%%
d_params, o_params, m_params, t_params, v_params, mca_params, mca_cls_params = load_params(
        "detector", "malgan", "metrics", "train", "valid", "mca", "mca_classifier")
#%%
class Experiment:
    @timing
    def __init__(self):
        self.notifier = Notifier(thread_name=t_params["name"])
        self.mca = mlflow.pyfunc.load_model(f"runs:/{mca_params['id']}/mca")
        self.__loadData()
        self.classifier = mlflow.sklearn.load_model(f"runs:/{mca_cls_params['id']}/mca_classifier")

    @log
    def __loadData(self, random_state=42):
        df_ben = pd.DataFrame(np.load("../" + v_params["benign"]))
        df_mal = pd.DataFrame(np.load("../" + v_params["malware"]))
        df_ben = self.mca.predict(df_ben)
        df_mal = self.mca.predict(df_mal)
        df_ben['class'] = 0  # Benign = 0
        df_mal['class'] = 1  # Malware = 1

        self.df = pd.concat([df_ben, df_mal], ignore_index=True)
        self.df = self.df.sample(
            frac=1, random_state=random_state).reset_index(drop=True)

    def __getY(self, keepObfuscated=False):
        return self.df['class'].transform(lambda x: 1 if x == 2 and not keepObfuscated else x)

    @property
    def X(self):
        return self.df.drop(columns=['class'])

    @property
    def y(self):
        return self.__getY()

    @property
    def y_obf(self):
        return self.__getY(keepObfuscated=True)

    @log
    @timing
    def run(self):
        y_pred = self.classifier.predict(self.X)
        report = classification_report(self.y, y_pred)
        confusion = confusion_matrix(self.y, y_pred, normalize='true')
        print(report)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=['Benign', 'Malware'])
        disp.plot().figure_.savefig("no_attack_mca.png")
        with open("no_attack_mca.json", "w") as f:
            f.write(self.metrics(y_pred))

    def metrics(self, y_pred, keepObfuscated=False):
        y_true = self.y if not keepObfuscated else self.y_obf
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        return json.dumps(metrics, indent=2)
# %%
exp = Experiment()
#%%
exp.run()