import json
import os

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def scale_numeric_data(books: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()
    books = books.copy()
    books[numeric_columns] = scaler.fit_transform(books[numeric_columns])
    return books[numeric_columns]


def description_distances(descriptions):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = np.abs(similarities - 1) / 2
    return np.array(similarities)


def preprocess(books: pd.DataFrame) -> pd.DataFrame:
    numeric_columns: list[str] = [
        "average_rating",
        "original_publication_year",
        "pages",
        "ratings_count",
        "genre_count",
    ]
    binary_columns: list[str] = json.load(open("./data/top_tags.json"))[:10]

    books["description_words"] = books["description_clean"].apply(
        lambda x: " ".join(list(set(x.split())))
    )

    df = books.loc[
        :, numeric_columns + binary_columns + ["description_clean", "description_words"]
    ]
    df[numeric_columns] = scale_numeric_data(df, numeric_columns)
    df[binary_columns] = df.loc[:, binary_columns]

    return df


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
    plt.show()


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

    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    return int(optimal_clusters)


def get_distance_matrix(df, descriptions, embeddings_weight=0.5):
    file_name = f"./data/distance_matrix_{embeddings_weight:.2f}.npy"
    if os.path.exists(file_name):
        return np.load(file_name)

    distance_matrix = gower.gower_matrix(df) * (1 - embeddings_weight)
    assert abs(embeddings_weight) <= 1
    if embeddings_weight == 0:
        np.save(file_name, distance_matrix)
        return distance_matrix

    distance_matrix += description_distances(descriptions) * embeddings_weight
    np.save(file_name, distance_matrix)
    return distance_matrix


def cluster_kmedoids(
    books,
    n_clusters=None,
    embeddings_weight=0.5,
    max_clusters=10,
    min_clusters=2,
):
    preprocessed = preprocess(books)
    distance_matrix = get_distance_matrix(
        preprocessed, books["description_clean"], embeddings_weight
    )
    print("Distance matrix calculated")

    if n_clusters is None:
        n_clusters = find_optimal_clusters_silhouette(
            distance_matrix, max_clusters=max_clusters, min_clusters=min_clusters
        )

    model = KMedoids(
        n_clusters=n_clusters,
        metric="precomputed",
        method="fasterpam",
        random_state=42,
        max_iter=1000,
    )
    books["cluster"] = model.fit_predict(distance_matrix)
    return books, distance_matrix


def cluster_hdbscan(
    books, min_samples=100, min_cluster_size=100, embeddings_weight=0.5
):
    preprocessed = preprocess(books)
    distance_matrix = get_distance_matrix(
        preprocessed, books["description_clean"], embeddings_weight
    )
    print("Distance matrix calculated")

    model = HDBSCAN(
        metric="precomputed", min_samples=min_samples, min_cluster_size=min_cluster_size
    )
    books["cluster"] = model.fit_predict(distance_matrix) + 2
    return books, distance_matrix
