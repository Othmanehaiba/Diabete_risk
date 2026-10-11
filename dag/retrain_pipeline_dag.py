from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'diabetes_risk_mlops_pipeline',
    default_args=default_args,
    description='Pipeline automatisée MLOps pour la prédiction du risque de diabète',
    schedule_interval='0 0 * * 0',  # Exécution hebdomadaire (dimanche à minuit)
    catchup=False,
) as dag:

    # Task 1 : Chargement et Prétraitement des Données
    preprocess_task = BashOperator(
        task_id='preprocess_data',
        bash_command='python -m src.data.preprocessing',
    )

    # Task 2 : Entraînement, Évaluation & Logging MLflow
    train_task = BashOperator(
        task_id='train_and_track_models',
        bash_command='python -m src.models.train_classifier',
    )

    # Task 3 : Test de l'API pour valider la prédiction
    healthcheck_api_task = BashOperator(
        task_id='verify_api_status',
        bash_command='curl -s -o /dev/null -w "%{http_code}" http://api:8000/ || exit 1',
    )

    # Dépendances du workflow
    preprocess_task >> train_task >> healthcheck_api_task