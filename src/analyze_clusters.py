import pandas as pd


def analyze_clusters(
    books, embeddings_weight: float = 0, interactions_weight: float = 0
):
    print(books.columns)
    books = books.drop(
        columns=["title", "description_clean", "description_words"]
    ).copy()

    grouped = books.groupby("cluster")

    numerical_columns = books.select_dtypes(include=["number"]).columns
    numerical_means = grouped[numerical_columns].mean()

    def combine_keywords(keywords_series):
        """Flatten lists in keywords and return unique values as a string."""
        all_keywords = []
        for keywords in keywords_series.dropna():
            all_keywords.extend(keywords if isinstance(keywords, list) else [keywords])
        return ", ".join(set(all_keywords))

    non_numerical_features = grouped.agg(
        {
            "keywords": combine_keywords,  # Handle keywords specifically
            "cluster_name": lambda x: x.mode()[0] if not x.mode().empty else "N/A",
        }
    )

    cluster_analysis = pd.concat([numerical_means, non_numerical_features], axis=1)
    cluster_analysis = cluster_analysis.drop(columns=["cluster"])
    cluster_analysis.to_csv(
        f"./output/analysis_{embeddings_weight:.2f}_{interactions_weight:.2f}.csv"
    )
