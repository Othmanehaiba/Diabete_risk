import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def init_mlflow_experiment(experiment_name: str = "Diabetes_Risk_Classification"):
    """Initialise le suivi MLflow avec SQLite."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

def log_run_to_mlflow(run_name, model, params, metrics, cm_path, X_sample, y_pred_sample):
    """Enregistre les paramètres, métriques, graphiques et la pipeline dans MLflow."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        if cm_path and os.path.exists(cm_path):
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            
        signature = infer_signature(X_sample, y_pred_sample)
        
        # Utilisation de cloudpickle pour sérialiser sans restriction imblearn et xgboost
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        
        return run.info.run_id