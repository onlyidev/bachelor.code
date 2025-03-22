r"""
    LIME verify - a component that uses LIME explanations to verify previous classification as benign
"""
import mlflow
import ast
import logging
from helpers.lime import LimeExplainer, CategoricalLimeExplainer
from functools import lru_cache
import numpy as np

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
    def __init__(self, run_ids, normal_features_path, mca_data_path, num_features=None):
        self.__run_ids = run_ids
        self.__loadModels()
        self.__loadNormalFeatures(normal_features_path)
        self.__initExplainer(mca_data_path)
        self.__num_features = len(self.__normal) if num_features == None else num_features
        
    def __loadModels(self):
        self.__mca = mlflow.pyfunc.load_model(f"runs:/{self.__run_ids['mca']}/mca")
        logger.info("Loaded MCA", extra={"run_id": self.__run_ids['mca']})
        self.__mca_classifier = mlflow.sklearn.load_model(f"runs:/{self.__run_ids['mca_cls']}/mca_classifier")
        logger.info("Loaded MCA Classifier", extra={"run_id": self.__run_ids['mca_cls']})
        
    def __loadNormalFeatures(self, path):
        with open(path, "r") as f:
            self.__normal = ast.literal_eval(f.read())
            logger.info(f"Loaded {len(self.__normal)} normal features")
            
    def __initExplainer(self, path):
        self.__explainer = LimeExplainer(path)
        logger.info("Loaded explainer")
         
    @lru_cache(maxsize=None)
    def transform(self, input: HashableType):
        logger.info("Transforming input to MCA")
        return self.__mca.predict(input.obj)
    
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
        exp = self.__explainer.explain(input.obj, self.__mca_classifier, num_features=self.__num_features)
        if outputResult:
            exp.show_in_notebook()
        features = set([name for name, _ in list(filter(lambda x: x[1] < 0,exp.as_list()))])
        isBenign = features.issubset(self.__normal)
        if not isBenign:
            logger.info("Non-standard features detected. Marking as malicious")
        return isBenign
    
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