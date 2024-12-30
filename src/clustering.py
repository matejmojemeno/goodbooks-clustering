import json
import os

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from .interactions import interactions


def scale_numeric_data(books: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()
    books = books.copy()
    books[numeric_columns] = scaler.fit_transform(books[numeric_columns])
    return books[numeric_columns]


def gower_distance_matrix(books: pd.DataFrame) -> np.ndarray:
    if os.path.exists("./data/gower_distance_matrix.npy"):
        return np.load("./data/gower_distance_matrix.npy")

    books = books.drop(columns=["description_clean", "book_id"])
    gower_matrix = gower.gower_matrix(books)

    np.save("./data/gower_distance_matrix.npy", gower_matrix)
    return gower_matrix


def description_distances(descriptions):
    if os.path.exists("./data/description_distances.npy"):
        return np.load("./data/description_distances.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = np.abs(similarities - 1) / 2

    np.save("./data/description_distances.npy", similarities)
    return np.array(similarities)


def preprocess(books: pd.DataFrame) -> pd.DataFrame:
    numeric_columns: list[str] = [
        "average_rating",
        "original_publication_year",
        "pages",
    ]
    binary_columns: list[str] = json.load(open("./data/top_tags.json"))[:10]

    df = books.loc[
        :, numeric_columns + binary_columns + ["description_clean", "book_id"]
    ]
    df[numeric_columns] = scale_numeric_data(df, numeric_columns)
    df[binary_columns] = df.loc[:, binary_columns]

    return df


def get_distance_matrix(
    df,
    embeddings_weight=0.33,
    interactions_weight=0.33,
):
    file_name = (
        f"./data/distance_matrix_{embeddings_weight:.2f}_{interactions_weight:.2f}.npy"
    )
    if os.path.exists(file_name):
        return np.load(file_name)

    gower_matrix = gower_distance_matrix(df) * (
        1 - embeddings_weight - interactions_weight
    )
    description_matrix = (
        description_distances(df["description_clean"]) * embeddings_weight
    )
    interactions_matrix = interactions(df) * interactions_weight

    distance_matrix = gower_matrix + description_matrix + interactions_matrix
    np.save(file_name, distance_matrix)
    return distance_matrix


def cluster(
    books,
    embeddings_weight=0.33,
    interactions_weight=0.33,
    model=KMedoids(n_clusters=10, metric="precomputed", method="fasterpam"),
):
    preprocessed = preprocess(books)
    distance_matrix = get_distance_matrix(
        preprocessed,
        embeddings_weight,
        interactions_weight,
    )
    print("Distance matrix calculated")

    books = books.drop(columns=["book_id"])

    print("Calculating clusters")
    books["cluster"] = model.fit_predict(distance_matrix)
    print("Clusters calculated")
    books["cluster"] += np.abs(books["cluster"].min())
    books["cluster"] = books["cluster"].astype(np.uint8)
    return books, distance_matrix
