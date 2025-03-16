import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
import dvc.api
from tqdm import tqdm


def aggregate_and_split_tensors(input_dir, output_file_train, output_file_valid, test_size=0.2, random_state=42, limit=None, num_features=None):
    """
    Aggregates tensors from multiple pickle files in a directory, splits the aggregated
    tensor into training and validation sets, and saves the resulting sets to separate
    pickle files.

    Args:
        input_dir (str): Path to the directory containing the pickle files.
        output_prefix (str): Prefix for the output file names.  The training set
            will be saved to "{output_prefix}_train.pkl" and the validation set
            to "{output_prefix}_val.pkl".
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random state for reproducible splitting.
    """
    tensors = []
    files = os.listdir(input_dir) if limit is None else os.listdir(
        input_dir)[:limit]
    for filename in tqdm(files, desc=f"Processing {input_dir}"):
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                tensor = pickle.load(f)
                tensors += tensor if num_features is None else tensor[:,
                                                                      :num_features]
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return
        except Exception as e:
            print(f"Error: Could not load tensor from {file_path}. {e}")
            return

    if not tensors:
        print("Error: No tensors found in the specified directory.")
        return

    # Aggregate the tensors
    try:
        aggregated_tensor = np.array(tensors)
    except Exception as e:
        print(f"Error: Could not aggregate the tensors. {e}")
        return

    # Split the aggregated tensor into training and validation sets
    try:
        X_train, X_val = train_test_split(
            aggregated_tensor, test_size=test_size, random_state=random_state)
    except Exception as e:
        print(f"Error: Could not split the aggregated tensor. {e}")
        return

    # Save the training set
    train_output_file = output_file_train
    try:
        with open(train_output_file, 'wb') as f:
            np.save(f, X_train)
        print(f"Training set saved to {train_output_file}")
    except Exception as e:
        print(
            f"Error: Could not save training set to {train_output_file}. {e}")
        return

    # Save the validation set
    val_output_file = output_file_valid
    try:
        with open(val_output_file, 'wb') as f:
            np.save(f, X_val)
        print(f"Validation set saved to {val_output_file}")
    except Exception as e:
        print(
            f"Error: Could not save validation set to {val_output_file}. {e}")
        return


if __name__ == "__main__":
    params = dvc.api.params_show()
    s_params = params["split"]
    t_params = params["train"]
    v_params = params["valid"]
    aggregate_and_split_tensors(s_params["benign_dir"], t_params["benign"], v_params["benign"], test_size=s_params["test_size"],
                                random_state=s_params["random_state"], limit=t_params["head"], num_features=t_params["num_features"])
    aggregate_and_split_tensors(s_params["malware_dir"], t_params["malware"], v_params["malware"], test_size=s_params["test_size"],
                                random_state=s_params["random_state"], limit=t_params["head"], num_features=t_params["num_features"])
