from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# Load and preprocess the data
def load_and_preprocess(file_path, genre_threshold=0.01):
    books = pd.read_csv(file_path)
    genres = books["genres"].apply(lambda x: x.split("'")[1::2])

    # Create a binary DataFrame for genres
    genres_set = set()
    for genre in genres:
        genres_set.update(genre)
    genres_set.remove("books")
    genres_set.remove("fiction")

    df = pd.DataFrame(
        {genre: books["genres"].apply(lambda x: genre in x) for genre in genres_set}
    )

    # Remove rare genres based on threshold
    min_count = genre_threshold * len(df)
    df = df.loc[:, df.sum() > min_count].astype(int)

    # Normalize the data
    normalized_df = normalize(df)

    return pd.DataFrame(normalized_df, columns=df.columns), books, genres_set


# Perform hierarchical clustering
def perform_clustering(data, method="ward", num_clusters=7):
    linkage_matrix = linkage(data, method=method)

    # Assign clusters
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion="maxclust")
    return clusters, linkage_matrix


# Automatically find the optimal number of clusters
def find_optimal_clusters(data, linkage_matrix, max_clusters=10):
    best_score = -1
    best_num_clusters = 2

    for n_clusters in range(2, max_clusters + 1):
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        score = silhouette_score(data, clusters)

        print(f"Number of clusters: {n_clusters}, Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_num_clusters = n_clusters

    print(f"\nOptimal number of clusters: {best_num_clusters}")
    return best_num_clusters


# Analyze clusters
def analyze_clusters(data, genres_set, clusters, n_clusters):
    data["cluster"] = clusters

    for cluster_id in range(1, n_clusters + 1):
        print(f"Cluster {cluster_id}:")
        cluster = data[data["cluster"] == cluster_id]
        print(f"Number of books: {cluster.shape[0]}")

        # Calculate genre prevalence in the cluster
        curr_genres = [
            cluster[cluster[genre] > 0].shape[0] / data[data[genre] > 0].shape[0]
            for genre in data.drop(columns=["cluster"]).columns
        ]

        while max(curr_genres) > 0.3:
            max_genre_index = curr_genres.index(max(curr_genres))
            max_genre = cluster.columns[max_genre_index]
            print(
                f"  {max_genre}: {curr_genres[max_genre_index]:.2f} {cluster[cluster[max_genre] > 0].shape[0]}"
            )
            curr_genres[max_genre_index] = -1


def assign_genres_to_clusters(data, genres_set, clusters, n_clusters):
    data["cluster"] = clusters

    cluster_genres = {i: [] for i in range(1, n_clusters + 1)}
    for genre in data.drop(columns=["cluster"]).columns:
        cluster_genre = []
        for cluster_id in range(1, n_clusters + 1):
            cluster = data[data["cluster"] == cluster_id]
            cluster_genre.append(cluster[cluster[genre] > 0].shape[0])

        cluster_genres[np.argmax(cluster_genre) + 1].append(
            (genre, np.sum(data[genre] > 0))
        )

    pprint(cluster_genres)

    for i in range(1, n_clusters + 1):
        book_set = set()
        for genre, _ in cluster_genres[i]:
            book_set.update(set(data[data[genre] > 0].index))
        print(f"Cluster {i}: {len(book_set)} books")


# Visualize clusters in 2D
def visualize_clusters(data, clusters, method="pca"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method: choose 'pca' or 'tsne'")

    reduced_data = reducer.fit_transform(data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap="viridis")
    plt.title(f"Clusters Visualized with {method.upper()}")
    plt.show()


# Main script
if __name__ == "__main__":
    file_path = "./goodbooks-10k/books_enriched.csv"

    # Step 1: Load and preprocess data
    df, books, genres_set = load_and_preprocess(file_path)

    # Step 2: Perform hierarchical clustering
    linkage_matrix = linkage(df, method="ward")

    # Step 3: Automatically find the optimal number of clusters
    max_clusters = 10
    optimal_clusters = find_optimal_clusters(df, linkage_matrix, max_clusters)
    # optimal_clusters = 5

    # Step 4: Perform clustering with the optimal number of clusters
    clusters, _ = perform_clustering(df, num_clusters=optimal_clusters)

    # Step 5: Analyze clusters
    analyze_clusters(df.copy(), genres_set, clusters, optimal_clusters)
    assign_genres_to_clusters(df.copy(), genres_set, clusters, optimal_clusters)

    # # Step 6: Visualize clusters
    # visualize_clusters(df, clusters, method="pca")
    visualize_clusters(df, clusters, method="tsne")
