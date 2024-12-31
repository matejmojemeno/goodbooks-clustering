import pandas as pd
from kmedoids import KMedoids

from src import (
    analyze_clusters,
    clean_dataset,
    cluster,
    name_clusters,
    visualize_clusters_umap,
)


def main():
    gower = True
    embeddings = True
    interact = True

    books = pd.read_csv("./goodbooks-10k/books.csv")
    print("Dataset loaded")

    books = clean_dataset(books)
    books = books.dropna().reset_index(drop=True)
    print("Dataset cleaned")

    model = KMedoids(n_clusters=15, metric="precomputed", method="fasterpam")
    books, distance_matrix = cluster(books, gower, embeddings, interact, model)

    books["description_words"] = books["description_clean"].apply(
        lambda x: " ".join(list(set(x.split())))
    )
    print(books["cluster"].value_counts())

    books = name_clusters(books)

    analyze_clusters(books, gower, embeddings, interact)

    visualize_clusters_umap(
        books,
        distance_matrix,
        gower,
        embeddings,
        interact,
        save=True,
    )

    print("Done: ", gower, embeddings, interact)


if __name__ == "__main__":
    main()
