import pandas as pd

from src import (
    analyze_clusters,
    clean_dataset,
    cluster_kmedoids,
    elbow_method,
    name_clusters,
    visualize_clusters,
    visualize_clusters_3d,
    visualize_clusters_umap,
)


def main():
    books = pd.read_csv("./goodbooks-10k/books.csv")
    print("Dataset loaded")
    books = clean_dataset(books)
    books = books.dropna().reset_index(drop=True)
    print("Dataset cleaned")

    embeddings_weight = 0.5
    books, distance_matrix = cluster_kmedoids(
        books, embeddings_weight=embeddings_weight, n_clusters=14, max_clusters=50
    )

    elbow_method(distance_matrix, n_clusters=14)
    exit()

    print("Dataset clustered")
    print(books["cluster"].value_counts())

    books = name_clusters(books)
    print("Clusters named")

    analyze_clusters(books, embeddings_weight)

    visualize_clusters(books, distance_matrix, embeddings_weight, save=True)
    visualize_clusters_3d(books, distance_matrix, embeddings_weight, save=True)
    print("Visualizations saved")


if __name__ == "__main__":
    main()
