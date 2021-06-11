import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from data_handler import get_data, get_headers
from ga import run_ga
from models import get_regression_models
import numpy as np
from kmeans_classifier import run_kmeans
from affinity_propagation_classifier import run_affinity_propagation
from custom_utils import print_on_file, get_metrics

np.random.seed(42)

X_train, x_test, Y_train, y_test = get_data()
FILENAME = './plots_10_folds/new_metrics.txt'

# CLUSTERS

run_kmeans(X_train, Y_train)
run_affinity_propagation(X_train, Y_train)

# REGRESSION MODELS


REGRESSION_MODELS = get_regression_models()
CV = StratifiedKFold(n_splits=10, shuffle=True)
scoring = ['accuracy',
           'precision_weighted',
           'recall_weighted',
           'f1_weighted']

print_on_file(text='\n## Regression Models Results ##', filename=FILENAME)

for model in REGRESSION_MODELS:

    train_scores = cross_validate(model['model'], X_train, Y_train, scoring=scoring, cv=CV, return_estimator=True)
    y_pred = train_scores['estimator'][-1].predict(x_test)

    print("Conteo de 1: " + str(np.count_nonzero(y_pred == 1)))
    print("Conteo de 0: " + str(np.count_nonzero(y_pred == 0)))

    report = classification_report(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = get_metrics(tn, fp, fn, tp)
    print_on_file(text='\nMODEL: ' + model['name'], filename=FILENAME)
    print_on_file(text=report, filename=FILENAME)
    print_on_file(text=metrics, filename=FILENAME)

    if model['name'] == 'Random Forest':
        fold_number = 1
        labels = get_headers()
        feature_importance_matrix = []
        for idx, estimator in enumerate(train_scores['estimator']):
            feature_importance_matrix.append(estimator.feature_importances_)
        feature_importances = np.mean(feature_importance_matrix, axis=0)

        print_on_file(text='Importancia de las variables: \n{}'.format(list(zip(labels, feature_importances))),
                      filename=FILENAME)
        plt.figure(figsize=(20, 8))
        figure = plt.barh(labels, feature_importances)
        plt.savefig('rf_importances.png')
        continue

    if model['name'] == 'Logistic Regression':
        coefs_matrix = []
        for estimator in train_scores['estimator']:
            coefs_matrix.append(estimator.coef_)
        coefs = np.mean(coefs_matrix, axis=0)
        labels = get_headers()
        plt.figure(figsize=(20, 8))
        plt.barh(labels, coefs[0])
        plt.savefig('lr_coefs.png')
        print_on_file(text='Coeficientes: \n{}'.format(list(zip(labels, coefs[0]))), filename=FILENAME)

# Neural Network + Genetic Algorithm

run_ga(X_train, Y_train)
