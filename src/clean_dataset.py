import os

import pandas as pd

from .description import get_cleaned_description
from .genres import split_genres


def clean_description(books):
    """Clean the description column for NLP processing."""
    books = get_cleaned_description(books)
    return books


def clean_genres(books):
    """Split the genres into multiple columns (one-hot encoding)."""
    books = split_genres(books)
    return books


def remove_columns(books):
    """Remove unnecessary columns from the dataset."""
    to_remove = [
        "goodreads_book_id",
        "best_book_id",
        "work_id",
        "books_count",
        "isbn",
        "isbn13",
        "authors",
        "authors_2",
        "original_title",
        "language_code",
        "work_ratings_count",
        "work_text_reviews_count",
        "ratings_1",
        "ratings_2",
        "ratings_3",
        "ratings_4",
        "ratings_5",
        "image_url",
        "small_image_url",
        "description",
        "index",
        "Unnamed: 0",
        "publishDate",
        "genres",
    ]

    books = books.drop(columns=to_remove)
    return books


def clean_dataset(books):
    """Clean the dataset for clustering."""
    if os.path.exists("./cache/books.csv"):
        return pd.read_csv("./cache/books.csv")

    books = clean_description(books)
    books = clean_genres(books)
    books = remove_columns(books)
    books["pages"] = books["pages"].fillna(books["pages"].median())
    books["original_publication_year"] = books["original_publication_year"].fillna(
        books["original_publication_year"].median()
    )
    books.to_csv("./cache/books.csv", index=False)
    return books
