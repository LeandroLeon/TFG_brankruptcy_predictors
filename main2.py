from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from custom_utils import get_metrics, print_on_file
import matplotlib.pyplot as plt
import logging
from data_handler import get_dataframe
import numpy as np
from k_fold_imblearn import KFoldImblearn
from models import get_regression_models


logging.getLogger('matplotlib.font_manager').disabled = True
np.random.seed(42)
FILENAME = 'testing_100folds.txt'

data = get_dataframe()

k_fold_imblearn_object = KFoldImblearn(
        sampling_method="RandomUnderSampler",
        k_folds=100,
        k_fold_shuffle=True,
        logging_level=10
)

k_fold_imblearn_object.k_fold_fit_resample(data.iloc[:, :-1], data.iloc[:, -1:], verbose=10, n_jobs=8)
dataset_list = k_fold_imblearn_object.k_fold_dataset_list

REGRESSION_MODELS = get_regression_models()
print_on_file(filename=FILENAME, text='\n## Regression Models Results ##')

total_cm = [0, 0, 0, 0]
for model in REGRESSION_MODELS:
    clf = model['model']
    model_name = model['name']
    x_test_accumulated = np.array([])
    y_test_accumulated = np.array([], dtype=np.int64)
    y_pred_accumulated = np.array([], dtype=np.int64)
    for fold in dataset_list:
        clf.fit(fold['resampled_train_set'][0], fold['resampled_train_set'][1])
        y_pred = clf.predict(fold['validation_set'][0])
        x_test_accumulated = np.concatenate((x_test_accumulated, fold['validation_set'][0].to_numpy())) if x_test_accumulated.size else fold['validation_set'][0].to_numpy()
        y_test_accumulated = np.concatenate((y_test_accumulated, fold['validation_set'][1].to_numpy())) if y_test_accumulated.size else fold['validation_set'][1].to_numpy()
        y_pred_accumulated = np.concatenate((y_pred_accumulated, y_pred)) if y_pred_accumulated.size else y_pred
    tn, fp, fn, tp = confusion_matrix(y_test_accumulated, y_pred_accumulated).ravel()
    print_on_file("MODEL: " + model_name, filename=FILENAME)
    print_on_file(classification_report(y_test_accumulated, y_pred_accumulated), filename=FILENAME)
    metrics = get_metrics(tn, fp, fn, tp)
    metrics['auc'] = roc_auc_score(y_test_accumulated, y_pred_accumulated)
    print(metrics)
    print_on_file(metrics, filename=FILENAME)
    #plot_roc_curve(clf, x_test_accumulated, y_test_accumulated)
    #plt.show()


