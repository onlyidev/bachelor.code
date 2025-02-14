# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow

# %% Load dataset
mlflow.enable_system_metrics_logging()
data_path = "data/dataset_malwares.csv"  # dataset_malwares.csv path
dataframe = pd.read_csv(data_path)
dataframe.drop(["Name"], axis=1, inplace=True)

# %% Correlation matrix to find the most important features

correlation_matrix = dataframe.corr()
target_correlation = correlation_matrix["Malware"].abs().sort_values(ascending=False)
important_features = [
    i for i in target_correlation.index if not pd.isna(target_correlation[i])
]
filtered_dataframe = dataframe[important_features]

# %% Normalization

scaler = MinMaxScaler()
normalized_df = filtered_dataframe.apply(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)

# %% Data split (40% for RF / 40% for MalGAN / 20% for testing)
malware = normalized_df[normalized_df["Malware"] == 1]
benign = normalized_df[normalized_df["Malware"] == 0]

malware = malware.drop(["Malware"], axis=1).values
benign = benign.drop(["Malware"], axis=1).values

SPLIT_PERCENT = 0.4
malware_split = int(SPLIT_PERCENT * malware.shape[0])
benign_split = int(SPLIT_PERCENT * benign.shape[0])

bb_malware, MalGAN_malware, test_malware = (
    malware[:malware_split],
    malware[malware_split : malware_split * 2],
    malware[malware_split * 2 :],
)
bb_benign, MalGAN_benign, test_benign = (
    benign[:benign_split],
    benign[benign_split : benign_split * 2],
    benign[benign_split * 2 :],
)

num_features = malware.shape[1]
noise_dim = num_features


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


blackBox, bb_cr = get_black_box(bb_malware, bb_benign)
print(bb_cr)

# %%
