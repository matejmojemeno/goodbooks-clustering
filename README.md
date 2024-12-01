# NI-MVI Semestral Work

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

## My approach
I will use Sentence Transformers to encode both user interaction data and book
metadata, creating a hybrid feature set for clustering. I will experiment with
HDBSCAN and k-means to identify meaningful groups, reflecting patterns in user
preferences and book characteristics. I will extract and analyze keywords from
genres and descriptions within each group to generate labels.


### Read articles
[SBERT paper](https://arxiv.org/pdf/1908.10084v1)


