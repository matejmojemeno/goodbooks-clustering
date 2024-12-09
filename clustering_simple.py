import json

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from kmedoids import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def preprocess_genres(
    books: pd.DataFrame, genre_groups: dict[str, list[str]]
) -> pd.DataFrame:
    books["genres"] = books["genres"].apply(lambda x: x.split("'")[1::2])

    for group in genre_groups:
        books[group] = books["genres"].apply(
            lambda x: len(set(x) & set(genre_groups[group])) > 0
        )
    books.drop(columns=["genres"], inplace=True)
    return books


def scale_numeric_data(books: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()
    books[numeric_columns] = scaler.fit_transform(books[numeric_columns])
    return books


def find_optimal_clusters_silhouette(distance_matrix, max_clusters=50):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        model = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            method="fasterpam",
            random_state=42,
        )
        labels = model.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        print(score)
        silhouette_scores.append(score)

    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    return int(optimal_clusters)


def cluster_kmedoids(distance_matrix, n_clusters=None):
    if n_clusters is None:
        n_clusters = find_optimal_clusters_silhouette(distance_matrix)

    model = KMedoids(
        n_clusters=n_clusters,
        metric="precomputed",
        method="fasterpam",
        random_state=42,
    )
    return model.fit_predict(distance_matrix)


def cluster_dbscan(distance_matrix, eps=0.07, min_samples=100):
    model = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    return model.fit_predict(distance_matrix) + 2


# def create_cluster_color_map(labels, min_cluster_size):
#     unique_clusters = np.unique(labels)
#     colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
#     cluster_to_color = {
#         cluster_id: plt.cm.tab20(idx) for idx, cluster_id in enumerate(unique_clusters)
#     }
#     return cluster_to_color


def animate_by_min_cluster_size(books, distance_matrix):
    colorscheme = {
        str(i): i / books["cluster"].nunique() for i in books["cluster"].unique()
    }

    min_cluster_sizes = range(1, 200, 5)

    tsne = TSNE(n_components=2, metric="precomputed", random_state=42, init="random")
    tsne_embedding = tsne.fit_transform(distance_matrix)

    animation_data = []
    for min_size in min_cluster_sizes:

        for cluster in books["cluster"].unique():
            cluster = books["cluster"] == cluster
            if cluster.sum() < min_size:
                books.loc[cluster, "cluster"] = 1

        df = pd.DataFrame(
            {
                "t-SNE 1": tsne_embedding[:, 0],
                "t-SNE 2": tsne_embedding[:, 1],
                "cluster": books["cluster"],
                "min_cluster_size": min_size,
            }
        )
        print(books["cluster"].nunique())
        animation_data.append(df)

    animation_data = pd.concat(animation_data, ignore_index=True)
    fig = px.scatter(
        animation_data,
        x="t-SNE 1",
        y="t-SNE 2",
        color="cluster",
        hover_data={"cluster": True},
        color_continuous_scale=colorscheme,
        animation_frame="min_cluster_size",
    )
    fig.show()


def visualize_clusters(books, distance_matrix):
    tsne = TSNE(n_components=2, metric="precomputed", random_state=42, init="random")
    tsne_embedding = tsne.fit_transform(distance_matrix)

    fig = px.scatter(
        books,
        x=tsne_embedding[:, 0],
        y=tsne_embedding[:, 1],
        color="cluster",
        hover_data={"cluster": True},
    )
    fig.show()


def main():
    books: pd.DataFrame = pd.read_csv("./goodbooks-10k/books_enriched.csv")
    books = books.dropna(subset=["description_clean"]).reset_index(drop=True)
    genre_groups: dict[str, list[str]] = json.load(open("./data/genre_groups.json"))
    books = preprocess_genres(books, genre_groups)

    numeric_columns: list[str] = [
        "average_rating",
        "books_count",
        "original_publication_year",
        "pages",
        "ratings_count",
    ]
    binary_columns: list[str] = list(genre_groups.keys())

    books = books.loc[:, numeric_columns + binary_columns]
    books["pages"] = books["pages"].fillna(books["pages"].median())
    books["original_publication_year"] = books["original_publication_year"].fillna(
        books["original_publication_year"].median()
    )

    books = scale_numeric_data(books, numeric_columns)
    books[binary_columns] = books[binary_columns].astype(int)

    distance_matrix = gower.gower_matrix(books)

    # books["cluster"] = cluster_kmedoids(distance_matrix, n_clusters=42)
    books["cluster"] = cluster_dbscan(distance_matrix, eps=0.07, min_samples=10).astype(
        np.uint64
    )

    # print(books["cluster"].value_counts())
    # visualize_clusters(books, distance_matrix)
    animate_by_min_cluster_size(books, distance_matrix)


if __name__ == "__main__":
    main()
