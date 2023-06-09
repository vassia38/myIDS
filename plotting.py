from scikitplot.estimators import plot_feature_importances
from scikitplot.metrics import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt

__RFC_NAME__ = 'Random Forest Classifier'

def show_feature_importance(estimator, X_feature_names):
    plot_feature_importances(estimator, feature_names=X_feature_names, max_num_features=15,
                                               figsize=(20,10), x_tick_rotation=45)
    plt.tight_layout(pad=5)
    plt.title("Feature Importance - " + __RFC_NAME__)
    plt.show()

def show_confusion_matrix(y_true, y_test, y_labels):
    plot_confusion_matrix(y_true, y_test, normalize=True,
                            x_tick_rotation=45, figsize=(20,10))
    plt.tight_layout(pad=5)
    plt.title("Confusion Matrix - " + __RFC_NAME__)
    plt.show()