import mlflow

def setupExperiment(name):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment = mlflow.get_experiment(mlflow.create_experiment(name))
    return experiment

def startExperiment(name = None, run_id = None, run_name = None):
    if run_id is not None:
        return mlflow.start_run(run_id=run_id, log_system_metrics=True)
    exp = setupExperiment(name)
    return mlflow.start_run(experiment_id=exp.experiment_id, run_name=run_name, log_system_metrics=True)

def logs():
    mlflow.autolog()
    mlflow.enable_system_metrics_logging()
    
    
def exportRunYaml(yamlFile="dynamic.yaml", key="experiment"):
    import yaml
    try:
        read = yaml.safe_load(open(yamlFile, "r"))
    except:
        read = dict()
    export = dict()
    export[key] = dict(id=mlflow.active_run().info.run_id)
    with open(yamlFile, "w") as f:
        yaml.dump(read|export, f)