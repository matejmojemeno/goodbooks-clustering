import json
import os

import pandas as pd

SUBSTITURIONS = {
    "ya": "young-adult",
    "non-fiction": "nonfiction",
    "sci-fi": "science-fiction",
}


TO_REMOVE = [
    "to-read",
    "fiction",
    "owned",
    "books-i-own",
    "series",
    "books-club",
    "default",
    "currently-reading",
    "favorites",
    "library",
    "book-club",
    "kindle",
    "ebook",
    "audiobook",
    "to-buy",
    "my-books",
    "audio",
    "novels",
    "owned-books",
    "favourites",
    "audiobooks",
    "ebooks",
]


def get_top_tags(books, n_genres) -> list:
    """Get the top n_genres tags from the dataset."""
    if os.path.exists("./cache/top_tags.json"):
        return json.load(open("./cache/top_tags.json"))[:n_genres]

    book_tags = pd.read_csv("./goodbooks-10k/book_tags.csv")
    tags = pd.read_csv("./goodbooks-10k/tags.csv")

    for key in SUBSTITURIONS:
        tags.tag_name = tags.tag_name.str.replace(key, SUBSTITURIONS[key])

    top_tags = set()
    counts = dict()

    for row in books["goodreads_book_id"]:
        curr_tags = book_tags[book_tags["goodreads_book_id"] == row]
        curr_tags = curr_tags.merge(tags, on="tag_id")

        for tag in TO_REMOVE:
            curr_tags = curr_tags[curr_tags["tag_name"] != tag]

        curr_tags = curr_tags.sort_values(by="count", ascending=False)

        top_tags.update(curr_tags["tag_name"][:15])

        for tag in curr_tags["tag_name"][:15]:
            if tag in counts:
                counts[tag] += 1
            else:
                counts[tag] = 1

    top_tags = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n_genres]
    top_tags = [tag for tag, _ in top_tags]

    json.dump(top_tags, open("./cache/top_tags.json", "w"))
    return top_tags
