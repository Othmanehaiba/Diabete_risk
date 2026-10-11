import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'risk_category'):
    """Isole les features (X) et la cible (y)."""
    cols_to_drop = [col for col in ['Cluster', 'risk_category', 'Outcome'] if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    return X, y

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Effectue la séparation Train/Test stratifiée."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)