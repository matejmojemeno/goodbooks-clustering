import numpy as np
import pandas as pd

from .top_tags import SUBSTITURIONS


def split_genres(books, genres):
    book_tags = pd.read_csv("./goodbooks-10k/book_tags.csv")
    tags = pd.read_csv("./goodbooks-10k/tags.csv")

    for genre in genres:
        books[genre] = np.int32(0)

    for row in books["goodreads_book_id"]:
        curr_tags = book_tags[book_tags["goodreads_book_id"] == row]
        curr_tags = curr_tags.merge(tags, on="tag_id")
        curr_tags = curr_tags.sort_values(by="count", ascending=False)[
            "tag_name"
        ].tolist()

        for genre in genres:
            if genre in curr_tags[:15]:
                books.at[row, genre] = np.int32(1)
            # elif genre in SUBSTITURIONS and SUBSTITURIONS[genre] in curr_tags[:15]:
            #     books.at[row, genre] = np.int32(1)

    for genre in genres:
        print(f"{genre}: {books[genre].sum()} books")
