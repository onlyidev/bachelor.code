r"""
    LIME verify - a component that uses LIME explanations to verify previous classification as benign
"""
import mlflow
import ast
import logging
from helpers.lime import LimeExplainer
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
class HashableType():
    def __init__(self, obj, key):
        self.obj = obj
        self.__key = key
    
    def __hash__(self):
        return hash(self.__key)
    
    def __eq__(self, other):
        return self.__key == other.__key

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
        self.__explainer = LimeExplainer(path)
        logger.info("Loaded explainer", extra={"run_id": self.__run_id})
         
    @lru_cache(maxsize=None)
    def transform(self, input: HashableType):
        logger.info("Transforming input to MCA", extra={"run_id": self.__run_id})
        return self.__mca.predict(input.obj)
    
    # Issue URL: https://github.com/onlyidev/bachelor.code/issues/3
    # milestone: Figure out LIME
    def verify(self, input: HashableType, outputResult=False):
        """Verifies if the input is benign (only meant to be used after initial classification)

        Args:
            input (feature vector): The input to be verified (should be transformed by MCA if required)
            outputResult (bool, optional): Show LIME explanation (in notebook). Defaults to False 

        Returns:
            bool: True if the input is benign, False otherwise
        """        
        exp = self.__explainer.explain_important(input.obj, self.__mca_classifier)
        if outputResult:
            exp.show_in_notebook()
        features = set([name for name, _ in exp.as_list()])
        isBenign = features.issubset(self.__normal)
        if not isBenign:
            logger.info("Non-standard features detected. Marking as malicious", extra={"run_id": self.__run_id})
        return isBenign