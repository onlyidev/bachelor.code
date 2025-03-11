import dvc.api

def load_params():
    params = dvc.api.params_show()
    return [params["experiment"], params["metrics"], params["split"], params["train"], params["valid"]]