import mlflow
import pickle
import os


class MCAWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["mca_model"], "rb") as f:
            self.mca = pickle.load(f)

    def predict(self, context, model_input):
        return self.mca.transform(model_input)


def log_model(run_id, model, artifact_path, input_example=None, registered_model_name=None):
    with open("mca_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=MCAWrapper(),
            artifacts={"mca_model": "mca_model.pkl"},
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
    os.remove("mca_model.pkl")
