from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def build_model_pipelines():
    """Définit les pipelines (Scaler + SMOTE + Classifieur) et leurs grilles d'hyperparamètres."""
    pipelines = {
        'LogisticRegression': ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', LogisticRegression(random_state=42))
        ]),
        'RandomForest': ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', RandomForestClassifier(random_state=42))
        ]),
        'XGBoost': ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
    }

    param_grids = {
        'LogisticRegression': {'model__C': [0.1, 1.0, 10.0]},
        'RandomForest': {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5, 10]},
        'XGBoost': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
    }
    
    return pipelines, param_grids