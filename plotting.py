import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

__RFC_NAME__ = 'Random Forest Classifier'

def show_feature_importance(estimator, X_feature_names, max_num_features=15):
    importances = estimator.feature_importances_
    feature_names = np.array(X_feature_names)
    indices = np.argsort(importances)[::-1][:max_num_features]
    top_features = feature_names[indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.barh(np.arange(len(top_features))[::-1], top_importances[::-1], align='center')
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance - ' + __RFC_NAME__)
    plt.tight_layout(pad=5)
    plt.show()


def show_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(18, 10))
    display.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('Confusion Matrix - ' + __RFC_NAME__)
    plt.tight_layout(pad=5)
    plt.show()