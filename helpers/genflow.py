import mlflow.pyfunc
import mlflow.pytorch
import torch


class CustomWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pt_model = mlflow.pytorch.load_model(
            context.artifacts["pytorch_model_uri"]
        )
        self.pt_model.eval()

    def predict(self, context, model_input):
        with torch.no_grad():
            output, _ = self.pt_model(model_input)
        return output


def log_model(model, artifact_path, **kwargs):
    artifacts = {"pytorch_model_uri": f"prep-{artifact_path}"}
    mlflow.pytorch.log_model(
        pytorch_model=model, artifact_path=artifacts["pytorch_model_uri"]
    )
    mlflow.pyfunc.save_model(
        path=artifact_path, python_model=CustomWrapper(), artifacts=artifacts, **kwargs
    )
