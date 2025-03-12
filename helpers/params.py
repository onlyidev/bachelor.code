import dvc.api

def load_params(*types) -> list[dict]:
    if not set(types).issubset({"experiment", "metrics", "split", "train", "valid"}):
        raise ValueError("Invalid parameter types")
    params = dvc.api.params_show()
    return [params[t] for t in types]