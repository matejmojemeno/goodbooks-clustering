import json
import os

import gower
import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

from .interactions import interactions


def scale_numeric_data(books: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Scale numeric columns to the range [0, 1]."""
    scaler = MinMaxScaler()
    books = books.copy()
    books[numeric_columns] = scaler.fit_transform(books[numeric_columns])
    return books[numeric_columns]


def gower_distance_matrix(books: pd.DataFrame) -> np.ndarray:
    """Compute the Gower distance matrix for the given DataFrame."""
    if os.path.exists("./cache/gower_distance_matrix.npy"):
        return np.load("./cache/gower_distance_matrix.npy")

    books = books.drop(columns=["description_clean", "book_id"])
    gower_matrix = gower.gower_matrix(books)

    np.save("./cache/gower_distance_matrix.npy", gower_matrix)
    return gower_matrix


def description_distances(descriptions):
    """Compute the distance matrix based on the descriptions of the books."""
    if os.path.exists("./cache/description_distances.npy"):
        return np.load("./cache/description_distances.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = np.abs(similarities - 1) / 2

    np.save("./cache/description_distances.npy", similarities)
    return np.array(similarities)


def preprocess(books: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset for clustering."""
    numeric_columns: list[str] = [
        "average_rating",
        "original_publication_year",
        "pages",
        "ratings_count",
    ]
    binary_columns: list[str] = json.load(open("./cache/top_tags.json"))[:10]

    df = books.loc[
        :, numeric_columns + binary_columns + ["description_clean", "book_id"]
    ]
    df[numeric_columns] = scale_numeric_data(df, numeric_columns)
    df[binary_columns] = df.loc[:, binary_columns]
    df = df.dropna()

    return df


def get_distance_matrix(
    df,
    gower=True,
    embeddings=True,
    interact=True,
):
    """Compute the distance matrix based on the given parameters."""
    distance_matrix = np.ones((df.shape[0], df.shape[0]))

    if gower:
        distance_matrix += gower_distance_matrix(df)
    if embeddings:
        distance_matrix += description_distances(df["description_clean"])
    if interact:
        distance_matrix += interactions(df)

    return distance_matrix


def cluster(
    books,
    gower=True,
    embeddings=True,
    interact=True,
    model=KMedoids(n_clusters=10, metric="precomputed", method="fasterpam"),
):
    """Cluster the books based on the given parameters."""
    preprocessed = preprocess(books)
    distance_matrix = get_distance_matrix(
        preprocessed,
        gower,
        embeddings,
        interact,
    )
    print("Distance matrix calculated")

    print("Calculating clusters")
    books["cluster"] = model.fit_predict(distance_matrix)
    print("Clusters calculated")
    books["cluster"] += np.abs(books["cluster"].min())
    books["cluster"] = books["cluster"].astype(np.uint8)
    return books, distance_matrix
