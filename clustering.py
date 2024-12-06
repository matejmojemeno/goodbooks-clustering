import json

import langid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, normalize


def load_data() -> pd.DataFrame:
    books: pd.DataFrame = pd.read_csv("./goodbooks-10k/books_enriched.csv")
    books.drop(
        columns=[
            "Unnamed: 0",
            "authors_2",
            "index",
            "image_url",
            "small_image_url",
            "isbn",
            "isbn13",
            "ratings_1",
            "ratings_2",
            "ratings_3",
            "ratings_4",
            "ratings_5",
            "language_code",
            "publishDate",
            "description",
            "authors",
            "best_book_id",
            "work_id",
            "goodreads_book_id",
            "book_id",
            "title",
            "original_title",
        ],
        inplace=True,
    )
    print("Loaded dataset, shape: ", books.shape)
    return books


def preprocess_genres(books: pd.DataFrame) -> pd.DataFrame:
    genre_groups = json.load(open("./data/genre_groups.json"))
    books["genres"] = books["genres"].apply(lambda x: x.split("'")[1::2])

    for group in genre_groups:
        books[group] = books["genres"].apply(
            lambda x: len(set(x) & set(genre_groups[group])) > 0
        )
    books.drop(columns=["genres"], inplace=True)
    # books.loc[:, genre_groups.keys()] = books.loc[:, genre_groups.keys()].apply(
    #     lambda x: x / x.sum(), axis=1
    # )

    return books


def create_embedding(books: pd.DataFrame) -> np.ndarray:
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = list(books["description_clean"])
    embeddings = sentence_transformer.encode(sentences)
    embeddings = normalize(embeddings)
    return embeddings


def dataset_for_clustering(books: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    numerical_features = books[["average_rating", "pages", "ratings_count"]]
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    genre_features = books[
        [
            "Young Adult",
            "Graphic Media",
            "Science and Fantasy",
            "Modern Fiction",
            "Literary Classics",
            "Historical Fiction",
            "Mystery and Thrillers",
            "Nonfiction",
        ]
    ].values

    embeddings = np.array(list(books["embeddings"]))

    dataset = np.hstack([scaled_numerical_features, genre_features, embeddings])
    return dataset


def main():
    books = load_data()
    books = books.dropna(subset=["description_clean"])
    books = preprocess_genres(books)
    embeddings = create_embedding(books)

    books["embeddings"] = list(embeddings)
    books.drop(columns=["description_clean"], inplace=True)
    books.dropna(inplace=True)

    dataset = dataset_for_clustering(books)
    print(dataset.shape)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(dataset)
    books["cluster"] = kmeans.labels_

    print(books["cluster"].value_counts())


if __name__ == "__main__":
    main()
