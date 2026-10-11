import os
import joblib
from sklearn.model_selection import GridSearchCV

from src.data.loader import load_raw_data
from src.data.preprocessing import prepare_features_and_target, split_data
from src.utils.metrics import compute_classification_metrics, save_confusion_matrix_plot
from src.models.pipeline import build_model_pipelines
from src.mlflow_utils.tracker import init_mlflow_experiment, log_run_to_mlflow

def main():
    # 1. Config MLflow
    init_mlflow_experiment()
    
    # 2. Chargement et Préparation
    data_path = "data/processed/df_classified_target.csv"
    if not os.path.exists(data_path):
        data_path = "../data/processed/df_classified_target.csv"
        
    df = load_raw_data(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 3. Pipelines & Modèles
    pipelines, param_grids = build_model_pipelines()
    
    best_global_f1 = 0
    champion_pipeline = None
    champion_run_id = None
    champion_name = ""

    # 4. Entraînement et MLflow Tracking
    for name, pipeline in pipelines.items():
        grid = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_pipeline = grid.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        
        metrics = compute_classification_metrics(y_test, y_pred)
        cm_path = save_confusion_matrix_plot(y_test, y_pred, name)
        
        run_id = log_run_to_mlflow(
            run_name=f"Run_{name}",
            model=best_pipeline,
            params=grid.best_params_,
            metrics=metrics,
            cm_path=cm_path,
            X_sample=X_train,
            y_pred_sample=y_pred
        )
        
        print(f"✅ {name} | F1-Score: {metrics['f1_score']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        
        if metrics['f1_score'] > best_global_f1:
            best_global_f1 = metrics['f1_score']
            champion_pipeline = best_pipeline
            champion_run_id = run_id
            champion_name = name

    # 5. Sauvegarde locale du modèle sous src/models/
    save_dir = "src/models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_diabetes_pipeline.joblib")
    
    joblib.dump(champion_pipeline, save_path)
    print(f"\n🏆 Champion Model : {champion_name} (F1: {best_global_f1:.4f})")
    print(f"✅ Modèle sauvegardé avec succès sous '{save_path}'")

if __name__ == "__main__":
    main()