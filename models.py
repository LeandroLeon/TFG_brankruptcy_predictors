from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC

MODELS = []


def add_model_with_name(clf, name):
    MODELS.append({'model': clf, 'name': name})


# NEURAL NETWORK
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(12, 5, 8), random_state=1, activation='relu', alpha=0.005,
                    max_iter=10000)
add_model_with_name(clf, "Neural Network")

# LOGISTIC REGRESSION
clf = LogisticRegression(solver='liblinear', penalty='l2')
add_model_with_name(clf, 'Logistic Regression')

# RANDOM FOREST
clf = RandomForestClassifier()
add_model_with_name(clf, 'Radom Forest')

SVM_KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']
SVC_GAMMAS = ['scale', 'auto']

# SVC
for kernel in SVM_KERNELS:
    for gamma in SVC_GAMMAS:
        clf = SVC(kernel=kernel, gamma=gamma)
        add_model_with_name(clf, 'SVC {} - {}'.format(kernel, gamma))

# NuSVC
for kernel in SVM_KERNELS:
    for gamma in SVC_GAMMAS:
        clf = NuSVC(kernel=kernel, gamma=gamma, nu=0.2)
        add_model_with_name(clf, 'NuSVC {} - {}'.format(kernel, gamma))

# GAUSSIAN NAIVE BAYES
clf = naive_bayes.GaussianNB()
add_model_with_name(clf, 'Gaussian NB')


# COMPLEMENT NAIVE BAYES
clf = naive_bayes.ComplementNB()
add_model_with_name(clf, 'Complement NB')


# Bagging Classifier
clf = BaggingClassifier(base_estimator=RandomForestClassifier())
add_model_with_name(clf, 'Ensemble Classifier')


def get_regression_models():
    return MODELS
