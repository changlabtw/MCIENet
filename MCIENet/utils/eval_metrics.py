"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2022.12.19
Last Update: 2023.10.16
"""
import functools
from sklearn import metrics

def get_evaluation(pred, label, pred_prob, multiclass:bool=False):
    result_dt = {k:v(y_true=label, y_pred=pred) for k, v in metrics_dt.items()}

    if multiclass:
        result_dt.update({k:v(y_true=label, y_score=pred_prob) for k, v in multiclass_prob_metrics_dt.items()})
    else:
        result_dt.update({k:v(y_true=label, y_score=pred_prob) for k, v in prob_metrics_dt.items()})

    report = metrics.classification_report(y_true=label, y_pred=pred, output_dict=True)

    return result_dt, report

# y_true, y_pred -> 類別, 類別 -----------------------------------------
metrics_dt = {
    'Precision': metrics.precision_score,
    'Recall': metrics.recall_score,
    'Weighted_Precision': functools.partial(metrics.precision_score, average='weighted'),
    'Balanced_acc': metrics.balanced_accuracy_score,
    'F1': functools.partial(metrics.f1_score, average='binary'),
    'matthews_corrcoef': metrics.matthews_corrcoef
}


# y_true, y_score -> 類別, 機率 -----------------------------------------
multiclass_prob_metrics_dt = {
    'Top_3_acc': functools.partial(metrics.top_k_accuracy_score, k=3),
    'Top_5_acc': functools.partial(metrics.top_k_accuracy_score, k=5),
    'Top_10_acc': functools.partial(metrics.top_k_accuracy_score, k=10),
    'Roc_auc_score(ovr)': functools.partial(metrics.roc_auc_score, multi_class='ovr'),
    'Roc_auc_score(ovo)': functools.partial(metrics.roc_auc_score, multi_class='ovo')
}

prob_metrics_dt = {
    'auPRCs': functools.partial(metrics.average_precision_score),
    'auROC': functools.partial(metrics.roc_auc_score)
}