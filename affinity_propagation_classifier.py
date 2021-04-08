from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from custom_utils import print_on_file


def run_affinity_propagation(X, Y):
    print_on_file(text='\n## Affinity Propagation Results ##\n')

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    DAMPING_OPTIONS = [0.5, 0.75, 0.9, 0.99]
    results = []

    for damping in DAMPING_OPTIONS:
        model = AffinityPropagation(damping=damping, random_state=42)
        model.fit(x_train)

        cluster_centers_indices = model.cluster_centers_indices_
        n_clusters_ = len(cluster_centers_indices)
        labels = model.labels_

        P = model.predict(x_train)

        silhouette_avg = silhouette_score(x_train, P)
        results.append({'damping': damping,
                        'silhouette_avg': silhouette_avg,
                        'n_clusters_': n_clusters_,
                        'model': model})

        print_on_file("For {} clusters and damping {}, the average silhouette_score is :{}".format(n_clusters_, damping,
                                                                                                      silhouette_avg))

    best_option = max(results, key=lambda k: k['silhouette_avg'])

    plt.plot()
    colors = cm.nipy_spectral(P.astype(float) / best_option['n_clusters_'])
    plt.scatter(x_train[:, 0], x_train[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = best_option['model'].cluster_centers_
    # Draw white circles at cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    plt.title("\nNª de clusters estimados: {} con damping = {}".format(best_option['n_clusters_'], best_option['damping']))

    plt.savefig('ap_plot.png')
    plt.show()
