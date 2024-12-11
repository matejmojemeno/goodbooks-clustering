import pandas as pd

from src.description import get_cleaned_description


def main():
    books = pd.read_csv("./goodbooks-10k/books.csv")
    books = get_cleaned_description(books)
