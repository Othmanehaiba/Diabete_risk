import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred):
    """Calcule l'ensemble des métriques d'évaluation."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred))
    }

def save_confusion_matrix_plot(y_true, y_pred, model_name: str, output_path: str = "confusion_matrix.png"):
    """Génère et sauvegarde l'image de la matrice de confusion."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.savefig(output_path)
    plt.close()
    return output_path