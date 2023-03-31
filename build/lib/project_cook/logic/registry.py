import glob
import os
import pickle
import time

from colorama import Fore, Style
from tensorflow import keras

from project_cook.params import *


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on mlflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params",
                                   timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics",
                                    timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it on your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on mlflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add breakpoint if you need!
        from google.cloud import storage

        model_filename = model_path.split("/")[
            -1]  # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to gcs")
        return None

    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model found

    """
    # get latest model version name by timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")
    if not local_model_paths:
        return None
    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add breakpoint if you need!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        from google.cloud import storage
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,
                                                 latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        lastest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return lastest_model

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
    model = keras.models.load_model(most_recent_model_path_on_disk)
    print("✅ model loaded from local disk")

    return model
