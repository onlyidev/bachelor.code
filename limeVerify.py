r"""
    LIME verify - a component that uses LIME explanations to verify previous classification as benign
"""
import mlflow
import logging
from helpers.lime import LimeExplainer, CategoricalLimeExplainer
from functools import lru_cache
import numpy as np
import pandas as pd
from helpers.params import load_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
class HashableType():
    def __init__(self, obj, key=None, compareByKey=True):
        self.obj = obj
        self.__key = key
        self.__compareByKey = compareByKey
    
    def __hash__(self):
        if isinstance(self.obj, np.ndarray):
            self.obj.flags.writeable = False
            return hash(self.obj.data.tobytes())
        return hash(self.__key)
    
    def __eq__(self, other):
        if self.__compareByKey:
            return self.__key == other.__key
        if isinstance(self.obj, np.ndarray):
            isEqual = np.array_equal(self.obj, other.obj)
            if isEqual:
                logger.debug(f"{self.obj} == {other.obj}")
            return isEqual
        return self.obj == other.obj

class LimeVerify:
    def __init__(self):
        t_params, mca_par, mca_cls_par = load_params("train", "mca", "mca_classifier")
        self.__mca_run_id = mca_par["id"]
        self.__mca_cls_run_id = mca_cls_par["id"]
        self.__loadModels()
        self.__loadNormalFeatures(t_params["normal_features"])
        self.__initExplainer(t_params["mca"])
        
    def __loadModels(self):
        self.mca = mlflow.pyfunc.load_model(f"runs:/{self.__mca_run_id}/mca")
        logger.info("Loaded MCA", extra={"run_id": self.__mca_run_id})
        self.mca_classifier = mlflow.sklearn.load_model(f"runs:/{self.__mca_cls_run_id}/mca_classifier")
        logger.info("Loaded MCA Classifier", extra={"run_id": self.__mca_cls_run_id})
        
    def __loadNormalFeatures(self, path):
        self.normal = pd.read_csv(path)
        logger.info(f"Loaded {len(self.normal)} normal features")
            
    def __initExplainer(self, path):
        self.explainer = LimeExplainer(path)
        logger.info("Loaded explainer")
         
    @lru_cache(maxsize=None)
    def transform(self, input: HashableType):
        logger.info("Transforming input to MCA")
        return self.mca.predict(input.obj)
    
    # Issue URL: https://github.com/onlyidev/bachelor.code/issues/3
    # milestone: Figure out LIME
    @lru_cache(maxsize=None)
    def verify(self, input: HashableType, outputResult=False):
        """Verifies if the input is benign (only meant to be used after initial classification)

        Args:
            input (feature vector): The input to be verified (should be transformed by MCA if required)
            outputResult (bool, optional): Show LIME explanation (in notebook). Defaults to False 

        Returns:
            bool: True if the input is benign, False otherwise
        """        
        exp = self.explainer.explain(input.obj, self.mca_classifier, num_features=int(np.sqrt(len(input.obj))))
        if outputResult:
            exp.show_in_notebook()
        features = pd.DataFrame([(k, -v if v < 0 else None) for k, v in exp.as_list()])
        features.columns = ["feature", "importance"]
        features = features.set_index("feature").join(self.normal.set_index("feature")).dropna() 
        features["deviation"] = (features["importance"] - features["average"]).abs()
        features["malicious"] = features["deviation"] > 3*features["std"]
        
        isMal = features["malicious"].any()
        if isMal:
            logger.info("Non-standard features detected. Marking as malicious")
        return not isMal
    
class CategoricalLimeVerify:
    def __init__(self, normal_features_path, run_id, num_features=1):
        self.__num_features = num_features
        self.__loadModels(run_id)
        self.__loadNormalFeatures(normal_features_path)
        self.__initExplainer()
        
    def __loadModels(self, run_id):
        self.__classifier = mlflow.sklearn.load_model(f"runs:/{run_id}/BB")
        
    def __loadNormalFeatures(self, path):
        with open(path, "r") as f:
            self.__normal = ast.literal_eval(f.read())
            logger.info(f"Loaded {len(self.__normal)} normal features")
            
    def __initExplainer(self):
        self.__explainer = CategoricalLimeExplainer()
        logger.info("Loaded explainer")
         
    # Issue URL: https://github.com/onlyidev/bachelor.code/issues/3
    # milestone: Figure out LIME
    @lru_cache(maxsize=None)
    def verify(self, input: HashableType, outputResult=False):
        """Verifies if the input is benign (only meant to be used after initial classification)

        Args:
            input (feature vector): The input to be verified (should be transformed by MCA if required)
            outputResult (bool, optional): Show LIME explanation (in notebook). Defaults to False 

        Returns:
            bool: True if the input is benign, False otherwise
        """        
        exp = self.__explainer.explain(input.obj, self.__classifier, num_features=self.__num_features)
        if outputResult:
            exp.show_in_notebook()
        features = set([name for name, _ in list(filter(lambda x: x[1] < 0,exp.as_list()))])
        isBenign = features.issubset(self.__normal)
        if not isBenign:
            logger.info("Non-standard features detected. Marking as malicious")
        return isBenign