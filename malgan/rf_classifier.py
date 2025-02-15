# %% Imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import malgan
import mlflow



# %% Train RF model
def get_black_box(malware, benign):
    """Trains a Random Forest black box classifier (for MalGAN) and registers it as a model

    Args:
        malware (pd_dataframe): Malware features
        benign (pd_dataframe): Benign (non-malware) features

    Returns:
        string, RandomForestClassifier: the classification report and the trained model
    """
    experiment = mlflow.get_experiment_by_name("MalGAN Black Box RF")
    with mlflow.start_run(experiment_id=experiment.experiment_id, log_system_metrics=True):
        mlflow.autolog()
        
        samples = np.concatenate((malware, benign), axis=0)
        labels = ([1] * malware.shape[0]) + ([0] * benign.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(
            samples, labels, test_size=0.2, random_state=0
        )

        rfc = RandomForestClassifier(
            n_estimators=50, random_state=0, oob_score=True, max_depth=8
        )
        history = rfc.fit(X_train, y_train)
        
        y_pred = rfc.predict(X_test)
        cr = classification_report(y_pred, y_test)
        
        mlflow.log_params(history.get_params())
        signature = mlflow.models.infer_signature(X_test, y_pred)
        
        mlflow.sklearn.log_model(rfc, "model", registered_model_name="MalGAN_BB_RF", signature=signature)
        return rfc, cr