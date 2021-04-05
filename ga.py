import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from random import randint
import random


def population_initialization_mlp(size_mlp):
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    pop = np.array([[random.choice(activation), random.choice(solver), randint(2, 100), randint(2, 50)]])
    for i in range(0, size_mlp - 1):
        pop = np.append(pop, [[random.choice(activation), random.choice(solver), randint(2, 50), randint(2, 50)]],
                        axis=0)
    return pop


def crossover(parent_1, parent_2):
    child = [parent_1[0], parent_2[1], parent_1[2], parent_2[3]]
    return child


def mutation(child, mut_prob):
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() >= mut_prob:
            k = randint(2, 3)
            child_[c, k] = int(child_[c, k]) + randint(1, 4)
    return child_


def fitness_function(pop, X_train, y_train, X_test, y_test):
    fitness = []
    j = 0
    for w in pop:
        clf = MLPClassifier(learning_rate_init=0.09, activation=w[0], solver=w[1], alpha=1e-5,
                            hidden_layer_sizes=(int(w[2]), int(w[3])), max_iter=1000, n_iter_no_change=80)

        try:
            clf.fit(X_train, y_train)
            f = accuracy_score(clf.predict(X_test), y_test)

            fitness.append([f, clf, w])
        except:
            pass
    return fitness  #


def ag_main(X_train, y_train, X_test, y_test, num_epochs=10, size_mlp=10, prob_mut=0.8):
    pop = population_initialization_mlp(size_mlp)
    fitness = fitness_function(pop, X_train, y_train, X_test, y_test)
    pop_fitness_sort = np.array(list(reversed(sorted(fitness, key=lambda x: x[0]))))

    for j in range(0, num_epochs):
        length = len(pop_fitness_sort)
        # Selecciona los padres
        parent_1 = pop_fitness_sort[:, 2][:length // 2]
        parent_2 = pop_fitness_sort[:, 2][length // 2:]

        # Cruzamiento
        child_1 = [crossover(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [crossover(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutation(child_2, prob_mut)

        # Calcula para ver que hijos pasan a la siguiente generacion
        fitness_child_1 = fitness_function(child_1, X_train, y_train, X_test, y_test)
        fitness_child_2 = fitness_function(child_2, X_train, y_train, X_test, y_test)
        pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
        sort = np.array(list(reversed(sorted(pop_fitness_sort, key=lambda x: x[0]))))

        # selecciona los individuos de la proxima generacion
        pop_fitness_sort = sort[0:size_mlp, :]
        best_individual = sort[0][1]


    return sort