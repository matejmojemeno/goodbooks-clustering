import matplotlib.pyplot as plt
from kmedoids import KMedoids


def elbow_method(distance_matrix, n_clusters=10):
    """Plot the elbow method to determine the optimal number of clusters."""
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
