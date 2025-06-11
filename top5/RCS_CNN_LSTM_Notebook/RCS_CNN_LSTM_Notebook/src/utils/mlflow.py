import mlflow

def setup_mlflow(config, experiment_name="RCS_Trading"):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=config.get("symbol", "default"))
    mlflow.log_params({
        "symbol": config.get("symbol"),
        "provider": config.get("provider"),
        "interval": config.get("interval"),
        "outputsize": config.get("outputsize"),
        "use_async": config.get("use_async"),
    })

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def end_mlflow():
    mlflow.end_run()
