from .description import get_cleaned_description
from .genres import split_genres


def clean_description(books):
    books = get_cleaned_description(books)
    books.to_csv("./data/books.csv", index=False)


def clean_genres(books):
    books = split_genres(books)
    books.to_csv("./data/books.csv", index=False)


def remove_columns(books):
    to_remove = [
        "Unnamed: 0",
        "book_id",
        "goodreads_book_id",
        "best_book_id",
        "work_id",
        "books_count",
        "isbn",
        "isbn13",
    ]

    print(books.columns)
