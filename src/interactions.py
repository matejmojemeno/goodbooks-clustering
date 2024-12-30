import os

import numpy as np
import pandas as pd

# def interactions(books):
#     if os.path.exists("./data/normalized_interaction_matrix.npy"):
#         return np.load("./data/normalized_interaction_matrix.npy")
#
#     ratings = pd.read_csv("./goodbooks-10k/ratings.csv")
#     ratings = ratings[ratings["book_id"].isin(books["book_id"])]
#     print(f"Number of unique books in ratings: {ratings['book_id'].nunique()}")
#
#     book_to_idx = {book_id: idx for idx, book_id in enumerate(books["book_id"])}
#     ratings["book_idx"] = ratings["book_id"].map(book_to_idx)
#
#     user_to_idx = {
#         user_id: idx for idx, user_id in enumerate(ratings["user_id"].unique())
#     }
#     ratings["user_idx"] = ratings["user_id"].map(user_to_idx)
#
#     user_book_matrix = np.zeros((ratings["user_idx"].nunique(), len(books)))
#     for _, row in ratings.iterrows():
#         user_book_matrix[row["user_idx"], row["book_idx"]] = 1
#
#     raw_interact_matrix = user_book_matrix.T @ user_book_matrix
#     book_counts = np.sum(user_book_matrix, axis=0)
#     norm_factors = np.minimum(book_counts[:, None], book_counts[None, :])
#
#     normalized_interact_matrix = np.divide(
#         raw_interact_matrix,
#         norm_factors,
#         out=np.zeros_like(raw_interact_matrix),
#         where=norm_factors > 0,
#     )
#
#     distance_matrix = 1 - normalized_interact_matrix
#     np.save("./data/normalized_interaction_matrix.npy", distance_matrix)
#     print("Normalized interaction matrix saved as a NumPy array.")
#
#     return distance_matrix


def interactions(books: pd.DataFrame):
    if os.path.exists("./data/normalized_interaction_matrix.npy"):
        return np.load("./data/normalized_interaction_matrix.npy")

    ratings = pd.read_csv("./goodbooks-10k/ratings.csv")
    ratings = ratings[ratings["book_id"].isin(books["book_id"])]
    print(f"Number of unique books in ratings: {ratings['book_id'].nunique()}")

    book_to_idx = {book_id: idx for idx, book_id in enumerate(books["book_id"])}
    ratings["book_idx"] = ratings["book_id"].map(book_to_idx)

    user_to_idx = {
        user_id: idx for idx, user_id in enumerate(ratings["user_id"].unique())
    }
    ratings["user_idx"] = ratings["user_id"].map(user_to_idx)

    user_book_matrix = np.zeros((ratings["user_idx"].nunique(), len(books)))
    for _, row in ratings.iterrows():
        user_book_matrix[row["user_idx"], row["book_idx"]] = 1

    # Compute the raw co-occurrence matrix
    raw_interact_matrix = user_book_matrix.T @ user_book_matrix

    # Compute the Jaccard similarity matrix
    book_counts = np.sum(user_book_matrix, axis=0)  # Number of users per book
    intersection = raw_interact_matrix
    union = (
        book_counts[:, None] + book_counts[None, :] - intersection
    )  # A ∪ B = |A| + |B| - |A ∩ B|

    jaccard_similarity_matrix = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )

    distance_matrix = 1 - jaccard_similarity_matrix
    np.save("./data/normalized_interaction_matrix.npy", distance_matrix)
    print("Jaccard similarity matrix saved as a distance matrix.")

    return distance_matrix
