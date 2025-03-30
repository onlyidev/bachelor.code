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
import warnings
from tqdm import tqdm
from scripts.notify import Notifier

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger()
tqdm.pandas(desc="Verifying")

class Experiment:
    @timing
    def __init__(self):
        self.notifier = Notifier(thread_name=t_params["name"])
        self.__loadData()
        self.detector = mlflow.sklearn.load_model(f"runs:/{d_params['id']}/BB")

    @log
    def __loadData(self, random_state=42):
        df_ben = pd.DataFrame(np.load(v_params["benign"]))
        df_mal = pd.DataFrame(np.load(v_params["malware"]))
        df_obf = self.__obfuscateData(df_mal)
        df_ben['class'] = 0  # Benign = 0
        df_mal['class'] = 1  # Malware = 1
        df_obf['class'] = 2  # Obfuscated Malware = 2

        self.df = pd.concat([df_ben, df_mal, df_obf], ignore_index=True)
        self.df = self.df.sample(
            frac=1, random_state=random_state).reset_index(drop=True)

    @log
    def __obfuscateData(self, data):
        gen = mlflow.pyfunc.load_model(f"runs:/{o_params['id']}/generator")
        return gen.predict(data)

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

    @abc.abstractmethod
    def run(self):
        pass

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


class NormalCase(Experiment):
    @log
    @timing
    def run(self):
        y_pred = self.detector.predict(self.X)
        report = classification_report(self.y, y_pred)
        confusion = confusion_matrix(self.y, y_pred, normalize='true')
        print(report)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=['Benign', 'Malware'])
        disp.plot().figure_.savefig(m_params["normal_confusion"])
        self.notifier.upload([m_params["normal_confusion"]], "Normal case")
        with open(m_params["normal"], "w") as f:
            f.write(self.metrics(y_pred))


class LimeCase(Experiment):

    @timing
    def __init__(self):
        super().__init__()
        self.verifier = limeVerify.LimeVerify()

    @log
    @timing
    def verify(self, preds, keepObfuscated=False):
        pdf = pd.DataFrame(preds)
        df = pdf[pdf[0] == 0]
        features = self.X[self.X.index.isin(df.index)].copy()
        t = self.verifier.transform(
            limeVerify.HashableType(features, "features"))
        v = t.progress_apply(lambda x: self.verifier.verify(
            limeVerify.HashableType(x.values, compareByKey=False)), axis=1)
        vt = v.transform(lambda x: 0 if x else 1).transform(
            lambda x: 2 if x == 1 and keepObfuscated else x)
        pdf.update(vt, overwrite=True)
        return pdf.values

    @log
    def printReports(self, y_true, y_pred, reportFile, confusionFile):
        report = classification_report(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred, normalize='true')
        print(report)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[
                                      'Benign', 'Malware'] if len(confusion) == 2 else ['Benign', 'Malware', 'Obfuscated'])
        disp.plot().figure_.savefig(confusionFile)
        with open(reportFile, "w") as f:
            f.write(self.metrics(y_pred))

    @log
    @timing
    def run(self):
        y_pred = self.detector.predict(self.X)  # Original prediction
        y_verified = self.verify(y_pred.copy())
        self.printReports(self.y, y_verified,
                          m_params["lime"], m_params["lime_confusion"])
        logger.info(self.verifier.transform.cache_info())
        logger.info(self.verifier.verify.cache_info())

        y_verified = self.verify(y_pred, keepObfuscated=True)
        self.printReports(self.y_obf, y_verified,
                          m_params["lime_obf"], m_params["lime_confusion_obf"])
        self.notifier.upload([m_params["lime_confusion_obf"]], "LIME (with MCA) case")
        logger.info(self.verifier.transform.cache_info())
        logger.info(self.verifier.verify.cache_info())

class LimeCategoricalCase(Experiment):
    @timing
    def __init__(self):
        super().__init__()
        self.verifier = limeVerify.CategoricalLimeVerify()

    @log
    def printReports(self, y_true, y_pred, reportFile, confusionFile):
        report = classification_report(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred, normalize='true')
        print(report)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[
                                      'Benign', 'Malware'] if len(confusion) == 2 else ['Benign', 'Malware', 'Obfuscated'])
        disp.plot().figure_.savefig(confusionFile)
        with open(reportFile, "w") as f:
            f.write(self.metrics(y_pred))
    
    @log
    @timing
    def verify(self, preds, keep_obfuscated=False):
        pdf = pd.DataFrame(preds)
        df = pdf[pdf[0] == 0]
        features = self.X[self.X.index.isin(df.index)].copy()
        v = features.progress_apply(lambda x: self.verifier.verify(
            limeVerify.HashableType(x.values, compareByKey=False)), axis=1) # type: ignore
        vt = v.transform(lambda x: 0 if x else 1).transform(
            lambda x: 2 if x == 1 and keep_obfuscated else x)
        pdf.update(vt, overwrite=True)
        return pdf.values
    
    @log
    @timing
    def run(self):
        y_pred = self.detector.predict(self.X)  # Original prediction
        y_verified = self.verify(y_pred.copy())
        self.printReports(self.y, y_verified,
                          m_params["lime_cat"], m_params["lime_cat_confusion"])
        logger.info(self.verifier.verify.cache_info())

        y_verified = self.verify(y_pred, keep_obfuscated=True)
        self.printReports(self.y_obf, y_verified,
                          m_params["lime_cat_obf"], m_params["lime_cat_confusion_obf"])
        self.notifier.upload([m_params["lime_cat_confusion_obf"]], "LIME (Fully Categorical) case")
        logger.info(self.verifier.verify.cache_info())

if __name__ == '__main__':
    d_params, o_params, m_params, t_params, v_params, mca_params, mca_cls_params = load_params(
        "detector", "malgan", "metrics", "train", "valid", "mca", "mca_classifier")
    args = sys.argv[1:]
    if len(args) < 1:
        raise ValueError("Please provide the experiment type")
    match args[0]:
        case "normal":
            logger.info("Running normal case")
            NormalCase().run()
        case "lime":
            logger.info("Running LIME case")
            LimeCase().run()
        case "categorical":
            logger.info("Running LIME Categorical case")
            LimeCategoricalCase().run()
        case _:
            raise ValueError("Invalid experiment type")
