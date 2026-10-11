import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset brut et nettoie les espaces dans les noms de colonnes."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df