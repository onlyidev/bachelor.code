import dvc.api

def load_params(*types) -> list[dict]:
    if not set(types).issubset({"malgan", "detector", "mca", "mca_classifier", "metrics", "split", "train", "valid"}):
        raise ValueError("Invalid parameter types")
    params = dvc.api.params_show()
    return [params[t] for t in types]