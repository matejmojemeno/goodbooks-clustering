import pandas as pd
from kmedoids import KMedoids
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    SpectralClustering,
)

from src import (
    analyze_clusters,
    clean_dataset,
    cluster,
    name_clusters,
    visualize_clusters,
    visualize_clusters_3d,
    visualize_clusters_umap,
)

models = {
    "KMedoids": KMedoids(
        n_clusters=20,
        metric="precomputed",
        method="pam",
        random_state=42,
        max_iter=1000,
    ),
    "DBSCAN": DBSCAN(
        eps=0.1485,
        min_samples=10,
        metric="precomputed",
    ),
    "HDBSCAN": HDBSCAN(
        min_samples=1,
        min_cluster_size=10,
        metric="precomputed",
    ),
    "SpectralClustering": SpectralClustering(
        n_clusters=10,
        affinity="precomputed",
        random_state=42,
    ),
    "AffinityPropagation": AffinityPropagation(
        affinity="precomputed",
    ),
    "AgglomerativeClustering": AgglomerativeClustering(
        n_clusters=10,
        metric="precomputed",
        linkage="average",
    ),
    "Birch": Birch(
        n_clusters=10,
        threshold=0.5,
    ),
}


def main():
    books = pd.read_csv("./goodbooks-10k/books.csv")
    print("Dataset loaded")
    books = clean_dataset(books)
    books = books.dropna().reset_index(drop=True)
    print("Dataset cleaned")

    embeddings_weight = 0.5
    interactions_weight = 0.1

    model = "KMedoids"
    # model = "DBSCAN"
    # model = "HDBSCAN"
    # model = "SpectralClustering"
    # model = "AffinityPropagation"
    # model = "AgglomerativeClustering"

    books, distance_matrix = cluster(
        books, embeddings_weight, interactions_weight, models[model]
    )

    books["description_words"] = books["description_clean"].apply(
        lambda x: " ".join(list(set(x.split())))
    )
    print(books["cluster"].value_counts())

    books = name_clusters(books)

    analyze_clusters(books, embeddings_weight, interactions_weight)

    visualize_clusters(
        books,
        distance_matrix,
        embeddings_weight,
        interactions_weight,
        save=True,
    )
    visualize_clusters_umap(
        books,
        distance_matrix,
        embeddings_weight,
        interactions_weight,
        save=True,
    )

    print("Done: ", embeddings_weight, interactions_weight)


if __name__ == "__main__":
    main()
