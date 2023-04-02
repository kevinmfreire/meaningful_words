'''
evaluate.py inludes functions:

Classification_summary --> For printing metric results base accuracy, F1, Precision and recall score.
                            it prints out the confusion matrix.

auc_roc_plot --> Plots the auc_roc for visualizing performance of ML model.
'''

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def classification_summary(pred, pred_prob, y_test, model):
    '''
    Prints out metrics for several models.
    Inputs: Model prediction, model confidence score, gorund truth, and model name
    Output: A classification summary based on accuracy, F1, Precision and recall score.
    '''
    print(
        "{}{}\033[1m Evaluating {} \033[0m{}{}\n".format(
            "<" * 3, "-" * 25, model, "-" * 25, ">" * 3
        )
    )
    print("Accuracy = {}%".format(round(accuracy_score(y_test, pred), 3) * 100))
    print(
        "F1 Score = {}%".format(
            round(
                f1_score(y_test, pred, average="weighted"), 3
                ) * 100
        )
    )
    print(
        "Precision Score = {}%".format(
            round(
                precision_score(y_test, pred, average="weighted"), 3
                ) * 100
        )
    )
    print(
        "Recall Score = {}%".format(
            round(
                recall_score(y_test, pred, average="weighted"), 3
                ) * 100
        )
    )

    print(
        "\n \033[1mConfusion Matrix:\033[0m\n", confusion_matrix(y_test, pred)
    )
    print(
        "\n\033[1mClassification Report:\033[0m\n", classification_report(y_test, pred)
    )


# Visualizing Function
def auc_roc_plot(y_test, pred):
    '''
    Plots the AUC-ROC plot.
    Inputs: ground truth data, and model prediction.
    Output: UC-ROC plot.
    '''
    ref = [0 for _ in range(len(y_test))]

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ref)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, pred)

    plt.plot(ns_fpr, ns_tpr, linestyle="=")
    plt.plot(
        lr_fpr,
        lr_tpr,
        marker="*",
        label="AUC = {}".format(round(roc_auc_score(y_test, pred) * 100, 2)),
    )
    plt.xlabel("Flase Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
