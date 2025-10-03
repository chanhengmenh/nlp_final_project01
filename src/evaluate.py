import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os

def evaluate_model(model, X_test, y_test, save_path=None):
    """Evaluate model and optionally save reports/plots."""
    y_pred = model.predict(X_test)

    # Generate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "report": report
    }

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save confusion matrix if requested
    if save_path:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join(save_path, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")

        # Save classification report as CSV
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(save_path, "classification_report.csv"))

    return results
