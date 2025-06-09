import optuna
import yaml
from train import train_model  # Assumes your train_model() returns accuracy/loss
from mlflow_utils import setup_mlflow, log_metrics, end_mlflow

def objective(trial):
    # Example parameter search space
    config = {
        "symbol": "EUR/USD",
        "provider": "twelvedata",
        "interval": "1min",
        "outputsize": 500,
        "use_async": True,
        "lstm_units": trial.suggest_int("lstm_units", 32, 128),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "window_size": trial.suggest_int("window_size", 10, 50),
        "epochs": 10
    }

    if config.get("mlflow_enabled", False):
        setup_mlflow(config)

    metrics = train_model(config)

    if config.get("mlflow_enabled", False):
        log_metrics(metrics)
        end_mlflow()

    return -metrics["loss"]  # Minimize loss

def run_optuna_study(n_trials=20, study_name="rcs_optuna", direction="minimize"):
    study = optuna.create_study(study_name=study_name, direction=direction)
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:", study.best_trial)

if __name__ == "__main__":
    run_optuna_study()
