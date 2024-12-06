import pandas as pd

books: pd.DataFrame = pd.read_csv("./goodbooks-10k/books_enriched.csv")
book_tags = pd.read_csv("./goodbooks-10k/book_tags.csv")
tags = pd.read_csv("./goodbooks-10k/tags.csv")


for i in range(5, 10):
    top_tags = set()
    counts = {}

    for row in books.goodreads_book_id:
        curr_tags = book_tags[book_tags.goodreads_book_id == row]
        curr_tags = curr_tags.merge(tags, on="tag_id")
        curr_tags = curr_tags.sort_values(by="count", ascending=False)
        top_tags.update(curr_tags.tag_name[:i])

        for tag in curr_tags.tag_name[:i]:
            if tag in counts:
                counts[tag] += 1
            else:
                counts[tag] = 1

    for tag in counts:
        if counts[tag] > 3000:
            top_tags.remove(tag)
            counts[tag] = 0
    print(len(top_tags))
    print(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])
