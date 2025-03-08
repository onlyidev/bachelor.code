r"""
    LIME verify - a component that uses LIME explanations to verify previous classification as benign
"""
import mlflow
import ast
import logging
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LimeVerify:
    
    def __init__(self, run_id, normal_features_path, mca_data_path):
        self.__run_id = run_id
        self.__loadModels()
        self.__loadNormalFeatures(normal_features_path)
        self.__initExplainer(mca_data_path)
        
    def __loadModels(self):
        self.__mca = mlflow.pyfunc.load_model(f"runs:/{self.__run_id}/mca")
        logger.info("Loaded MCA", extra={"run_id": self.__run_id})
        self.__mca_classifier = mlflow.sklearn.load_model(f"runs:/{self.__run_id}/mca_classifier")
        logger.info("Loaded MCA Classifier", extra={"run_id": self.__run_id})
        
    def __loadNormalFeatures(self, path):
        with open(path, "r") as f:
            self.__normal = ast.literal_eval(f.read())
            logger.info(f"Loaded {len(self.__normal)} normal features", extra={"run_id": self.__run_id})
            
    def __initExplainer(self, path):
        mca_data = pd.read_csv(path)
        self.__explainer = LimeTabularExplainer(mca_data.values[:,:-1], feature_names=mca_data.columns[:-1], class_names=['Benign', 'Malicious'], verbose=True, sample_around_instance=True)
        logger.info("Loaded explainer", extra={"run_id": self.__run_id})
            
    def transform(self, input):
        return self.__mca.predict(input)
    
    # TODO Determine for which class the explanations are meant
    # milestone: Figure out LIME
    def verify(self, input, outputResult=False):
        exp = self.__explainer.explain_instance(input, self.__mca_classifier.predict_proba, labels=(0,))
        if outputResult:
            exp.show_in_notebook()
        # features = set([name for name, _ in filter(lambda pair: pair[1] > 0, exp.as_list())])
        # isMal = not features.issubset(self.__normal)
        # print(isMal)
        # if isMal:
        #     logger.info("Non-standard features detected. Marking as malicious", extra={"run_id": self.__run_id})
        return exp