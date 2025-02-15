#%% Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
# import rf_classifier
import malgan
import mlflow

#%% Config
mlflow.enable_system_metrics_logging()
mlflow.config.enable_async_logging()

# %% Load dataset
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
print(num_features, noise_dim)

# %% Handle batching
BATCH_SIZE = malgan.BATCH_SIZE
malware_shape = (MalGAN_malware.shape[0] // BATCH_SIZE, BATCH_SIZE, MalGAN_malware.shape[1])
benign_shape = (MalGAN_benign.shape[0] // BATCH_SIZE, BATCH_SIZE, MalGAN_benign.shape[1])

MalGAN_malware = MalGAN_malware[: (MalGAN_malware.shape[0] // BATCH_SIZE) * BATCH_SIZE].reshape(malware_shape)
MalGAN_benign = MalGAN_benign[: (MalGAN_benign.shape[0] // BATCH_SIZE) * BATCH_SIZE].reshape(benign_shape)

#%% Get RF classifier

# blackBox, bb_cr = rf_classifier.get_black_box(bb_malware, bb_benign)
# print(bb_cr)
logged_model = 'runs:/043b170ccf6c402e9fc7851e6725b99d/model'
blackBox = mlflow.pyfunc.load_model(logged_model)

#%% Training MalGAN
generator, substituteDetector, gan = malgan.getMalGAN(num_features)
experiment = mlflow.get_experiment_by_name("MalGAN Adversarial Attack Generator")
with mlflow.start_run(experiment_id=experiment.experiment_id, log_system_metrics=True):
    malgan.train(generator, blackBox, substituteDetector, gan, MalGAN_malware, MalGAN_benign, epochs=10)
    
    test_samples, test_labels = malgan.getTestData(generator, test_malware, test_benign)
    y_pred = blackBox.predict(test_samples)
    report = classification_report(y_pred, test_labels)
    print(report)
    
    signature = mlflow.models.infer_signature(test_samples, y_pred)
    mlflow.keras.log_model(generator, "model/gen", registered_model_name="MALGAN_ADV_GENERATOR", signature=signature)
    
#%% TF test
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))