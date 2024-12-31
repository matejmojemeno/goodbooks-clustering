# The Goodbooks Dataset Clustering

## The assignment
The "goodbooks" dataset contains six million ratings for the ten
thousand most popular books. It includes data such as books marked
"to-read" by users, book metadata (author, year, etc.), and
tags/shelves/genres. For this task, we would like you to cluster similar
books by considering both interaction data (user ratings) and metadata.
Although the dataset includes some metadata (e.g., author and title),
feel free to collect additional data from the web or generate new data.
Additionally, please propose a method to automatically label the
clusters in a way that reflects their semantic content. We encourage
creativity in this process and are not setting strict guidelines for
clustering.

## Where to gather data

The following files are needed from the `goodbooks-10k` dataset accessible on [GitHub](https://github.com/zygmuntz/goodbooks-10k):
- `ratings.csv`
- `tags.csv`
- `book_tags.csv`

The file `books_enriched.csv` is needed from the `goodbooks-10k-extended` dataset accessible on [GitHub](https://github.com/malcolmosh/goodbooks-10k-extended)
The files need to be placed in the `goodbooks-10k` directory in the root of the project.

## How to run the code
The project is a python project, and can be run by simply running the `main.py` file.
The output of the project is a `csv` file containing the clusters and their labels, and an interactive `html` visualization of the clusters.

## The Visualizations:
- [Without semantic content](https://matejmojemeno.github.io/clustering_visualization/umap_gower.html)

- [With semantic content](https://matejmojemeno.github.io/clustering_visualization/umap_combined.html)
