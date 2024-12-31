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
    scaler = MinMaxScaler()
    books = books.copy()
    books[numeric_columns] = scaler.fit_transform(books[numeric_columns])
    return books[numeric_columns]


def gower_distance_matrix(books: pd.DataFrame) -> np.ndarray:
    if os.path.exists("./cache/gower_distance_matrix.npy"):
        return np.load("./cache/gower_distance_matrix.npy")

    books = books.drop(columns=["description_clean", "book_id"])
    gower_matrix = gower.gower_matrix(books)

    np.save("./cache/gower_distance_matrix.npy", gower_matrix)
    return gower_matrix


def description_distances(descriptions):
    if os.path.exists("./cache/description_distances.npy"):
        return np.load("./cache/description_distances.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    similarities = model.similarity(embeddings, embeddings)
    similarities = np.abs(similarities - 1) / 2

    np.save("./cache/description_distances.npy", similarities)
    return np.array(similarities)


def preprocess(books: pd.DataFrame) -> pd.DataFrame:
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
    file_name = (
        f"./cache/distance_matrix_{int(gower)}_{int(embeddings)}_{int(interact)}.npy"
    )
    # if os.path.exists(file_name):
    #     return np.load(file_name)

    distance_matrix = np.ones((df.shape[0], df.shape[0]))

    if gower:
        distance_matrix += gower_distance_matrix(df)
    if embeddings:
        distance_matrix += description_distances(df["description_clean"])
    if interact:
        distance_matrix += interactions(df)

    np.save(file_name, distance_matrix)
    return distance_matrix


def cluster(
    books,
    gower=True,
    embeddings=True,
    interact=True,
    model=KMedoids(n_clusters=10, metric="precomputed", method="fasterpam"),
):
    preprocessed = preprocess(books)
    distance_matrix = get_distance_matrix(
        preprocessed,
        gower,
        embeddings,
        interact,
    )
    print("Distance matrix calculated")

    # books = books.drop(columns=["book_id"])

    print("Calculating clusters")
    books["cluster"] = model.fit_predict(distance_matrix)
    print("Clusters calculated")
    books["cluster"] += np.abs(books["cluster"].min())
    books["cluster"] = books["cluster"].astype(np.uint8)
    return books, distance_matrix
