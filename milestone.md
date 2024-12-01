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

- [Sentence Transformers: Sentence Embeddings using BERT](https://arxiv.org/pdf/1908.10084v1)
- [K-Means and Hierarchical clustering on the GoodReads dataset](https://medium.com/@pertchuhi_proshyan/k-means-and-hierarchical-clustering-on-the-goodreads-dataset-18c346e33566) 
- [Short text clustering with Sentence Transformers](https://arxiv.org/abs/2102.00541)
- [Improving the Performance of K-Means Clustering for High Dimensional Data Set](https://www.researchgate.net/profile/P-Prabhu/publication/278401467_Improving_Business_Intelligence_Based_on_Frequent_Itemsets_Using_k-Means_Clustering_Algorithm/links/5c4bf06692851c22a3911bbc/Improving-Business-Intelligence-Based-on-Frequent-Itemsets-Using-k-Means-Clustering-Algorithm.pdf)

### Current progress
- I found an [extended version of the goodbooks dataset](https://github.com/malcolmosh/goodbooks-10k-extended) that includes additional metadata such as book descriptions and genres. I will use this dataset to create a hybrid feature set for clustering.
- I also used the description feature to identify the language of the book and filter out non-English books, since they are a small minority in the dataset.
- I tried to use one-hot encoding for genres for the clustering, but this creates a sparse matrix that might not be suitable for clustering. (There is 39 unique genres in the dataset)
- I am trying to group the genres into broader categories to reduce the dimensionality of the feature set. I will be doing this by clustering the genres without the user interaction data and then assigning each genre to the cluster with the most similar genres.

### Current results
- I spent most of the time cleaning the data and preparing the feature set for clustering. The first attempt at clustering with k-means did not yield meaningful results.
