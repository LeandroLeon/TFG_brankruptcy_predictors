from sklearn.cluster import AffinityPropagation, KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, learning_curve
from data_handler import get_data, get_headers
from ga import ag_main
from models import get_models
import numpy as np

np.random.seed(42)

X, Y = get_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
MODELS = get_models()
CV = StratifiedKFold(n_splits=10, shuffle=True)
scoring = ['accuracy',
           'precision_weighted',
           'recall_weighted',
           'f1_weighted']

"""
for model in MODELS:
    resultsFile = open('results.txt', 'a+')
    scores = cross_validate(model['model'], X, Y, scoring=scoring, cv=CV, return_estimator=True)
    print('\nMODEL: ' + model['name'], file=resultsFile)
    print('Accuracy: %.4f (%.4f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])), file=resultsFile)
    print('Precision: %.4f (%.4f)' % (np.mean(scores['test_precision_weighted']), np.std(scores['test_precision_weighted'])), file=resultsFile)
    print('Recall: %.4f (%.4f)' % (np.mean(scores['test_recall_weighted']), np.std(scores['test_recall_weighted'])), file=resultsFile)
    print('F1 Score: %.4f (%.4f)' % (np.mean(scores['test_f1_weighted']), np.std(scores['test_f1_weighted'])), file=resultsFile)

    if(model['name'] == 'Logistic Regression'):
        coefs_matrix = []
        for estimator in scores['estimator']:
            coefs_matrix.append(estimator.coef_)
        coefs = np.mean(coefs_matrix, axis=0)
        labels = get_headers()
        print('Coeficientes: \n {}'.format(list(zip(labels, coefs[0]))), file=resultsFile)
    resultsFile.close()
# AFFINITY PROPAGATION

model = AffinityPropagation(damping=0.99, random_state=42)
model.fit(x_train)

cluster_centers_indices = model.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
labels = model.labels_

P = model.predict(x_train)

silhouette_avg = silhouette_score(x_train, P)
resultsFile = open('results.txt', 'a+')
print("\nFor {} clusters, the average silhouette_score is :{}".format(n_clusters_, silhouette_avg), file=resultsFile)
resultsFile.close()

plt.plot()
colors = cm.nipy_spectral(P.astype(float) / n_clusters_)
plt.scatter(x_train[:, 0], x_train[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = model.cluster_centers_
# Draw white circles at cluster centers
plt.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

plt.title("\nNª de clusters estimados: {}".format(n_clusters_))

plt.show()


# K-MEANS
clusterer = KMeans(n_clusters=2, random_state=0)
cluster_labels = clusterer.fit_predict(x_train)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters


plt.plot()
colors = cm.nipy_spectral(cluster_labels.astype(float) / 2)
plt.scatter(x_train[:, 0], x_train[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
plt.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

plt.title("Visualización de K-Means; nº de clusters = 2")

plt.show()
"""

# Neural Network + Genetic Algorithm

best_result = ag_main(x_train, y_train, x_test, y_test, num_epochs=10, size_mlp=20, prob_mut=0.9)
print(best_result)
scores = cross_validate(best_result[0][1], X, Y, scoring=scoring, cv=CV)
resultsFile = open('results.txt', 'a+')
print('\nMODEL: Red Neuronal + Algortimo Genético', file=resultsFile)
print('Accuracy: %.4f (%.4f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])), file=resultsFile)
print('Precision: %.4f (%.4f)' % (np.mean(scores['test_precision_weighted']), np.std(scores['test_precision_weighted'])), file=resultsFile)
print('Recall: %.4f (%.4f)' % (np.mean(scores['test_recall_weighted']), np.std(scores['test_recall_weighted'])), file=resultsFile)
print('F1 Score: %.4f (%.4f)' % (np.mean(scores['test_f1_weighted']), np.std(scores['test_f1_weighted'])), file=resultsFile)
resultsFile.close()

fig, ax = plt.subplots()

train_sizes, train_scores, test_scores = \
    learning_curve(best_result, x_train, y_train, cv=None, n_jobs=None,
                   train_sizes=np.linspace(.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

ax.set(xlabel='Training examples', ylabel='Score',
       title='Red Neuronal + Algoritmo Genético')
ax.grid()
ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.show()
