import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from data_handler import get_data, get_headers
from ga import run_ga
from models import get_regression_models
import numpy as np
from kmeans_classifier import run_kmeans
from affinity_propagation_classifier import run_affinity_propagation
from custom_utils import print_on_file, format_scores

np.random.seed(42)

X, Y = get_data()

# CLUSTERS

run_kmeans(X, Y)
run_affinity_propagation(X, Y)

# REGRESSION MODELS

REGRESSION_MODELS = get_regression_models()
CV = StratifiedKFold(n_splits=10, shuffle=True)
scoring = ['accuracy',
           'precision_weighted',
           'recall_weighted',
           'f1_weighted']

print_on_file(text='\n## Regression Models Results ##')

for model in REGRESSION_MODELS:

    scores = cross_validate(model['model'], X, Y, scoring=scoring, cv=CV, return_estimator=True)
    print_on_file(text='\nMODEL: ' + model['name'])
    print_on_file(text=format_scores(scores))

    if model['name'] == 'Random Forest':
        fold_number = 1
        labels = get_headers()
        feature_importance_matrix = []
        for idx, estimator in enumerate(scores['estimator']):
            feature_importance_matrix.append(estimator.feature_importances_)
        feature_importances = np.mean(feature_importance_matrix, axis=0)

        print_on_file(text='Importancia de las variables: \n{}'.format(list(zip(labels, feature_importances))))
        figure = plt.barh(labels, feature_importances)
        plt.savefig('rf_importances.png')
        continue

    if model['name'] == 'Logistic Regression':
        coefs_matrix = []
        for estimator in scores['estimator']:
            coefs_matrix.append(estimator.coef_)
        coefs = np.mean(coefs_matrix, axis=0)
        labels = get_headers()
        plt.barh(labels, coefs[0])
        plt.savefig('lr_coefs.png')
        print_on_file(text='Coeficientes: \n{}'.format(list(zip(labels, coefs[0]))))

# Neural Network + Genetic Algorithm

run_ga(X, Y)
