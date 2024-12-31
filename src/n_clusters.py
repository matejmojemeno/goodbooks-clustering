import matplotlib.pyplot as plt
import numpy as np
from kmedoids import KMedoids
from sklearn.metrics import silhouette_score


def elbow_method(distance_matrix, n_clusters=10):
    sum_of_squared_distances = []
    K = range(1, 50)
    for k in K:
        model = KMedoids(
            n_clusters=k,
            metric="precomputed",
            method="fasterpam",
            random_state=42,
            max_iter=1000,
        )
        model.fit(distance_matrix)
        sum_of_squared_distances.append(model.inertia_)

    plt.plot(K, sum_of_squared_distances, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum of squared distances")
    plt.title("Elbow Method For Optimal k")
    plt.axvline(x=n_clusters, color="r", linestyle="--")
    plt.savefig("./output/elbow_method.png")


def find_optimal_clusters_silhouette(distance_matrix, max_clusters=10, min_clusters=2):
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        model = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            method="fasterpam",
            random_state=42,
            max_iter=1000,
        )
        labels = model.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        print(n_clusters, score)
        silhouette_scores.append(score)

    optimal_clusters = np.argmax(silhouette_scores) + 2
    return int(optimal_clusters)
