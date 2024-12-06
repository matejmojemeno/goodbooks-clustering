import json

import langid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


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
        ],
        inplace=True,
    )
    print("Loaded dataset, shape: ", books.shape)
    return books


def filter_non_english(books: pd.DataFrame) -> pd.DataFrame:
    books.dropna(subset=["description"], inplace=True)
    books["language"] = books["description"].apply(lambda x: langid.classify(x)[0])
    books = books[books["language"] == "en"].drop(columns=["language"])
    print("Filtered non-english books, shape: ", books.shape)
    return books


def preprocess_genres(books: pd.DataFrame) -> pd.DataFrame:
    genre_groups = json.load(open("./data/genre_groups.json"))
    books["genres"] = books["genres"].apply(lambda x: x.split("'")[1::2])

    for group in genre_groups:
        books[group] = (
            books["genres"]
            .apply(lambda x: 1 if len(set(x) & set(genre_groups[group])) > 0 else 0)
            .astype("float")
        )
    books.drop(columns=["genres"], inplace=True)
    books.loc[:, genre_groups.keys()] = books.loc[:, genre_groups.keys()].apply(
        lambda x: x / x.sum(), axis=1
    )

    return books


def main():
    books = load_data()
    books = filter_non_english(books)
    books = preprocess_genres(books)

    genre_groups = json.load(open("./data/genre_groups.json"))
    print(books.loc[:, genre_groups.keys()].head())


if __name__ == "__main__":
    main()
