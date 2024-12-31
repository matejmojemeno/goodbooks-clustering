import pandas as pd


def combine_keywords(keywords_series: pd.Series) -> str:
    """Flatten lists in keywords and return unique values as a string."""
    all_keywords = []
    for keywords in keywords_series.dropna():
        all_keywords.extend(keywords if isinstance(keywords, list) else [keywords])
    return ", ".join(set(all_keywords))


def analyze_clusters(
    books: pd.DataFrame, gower: bool, embeddings: bool, interact: bool
) -> None:
    """Analyze clusters and save results to a CSV file."""
    books = books.drop(
        columns=["title", "description_clean", "description_words"]
    ).copy()

    grouped: pd.DataFrameGroupBy = books.groupby("cluster")

    numerical_columns = books.select_dtypes(include=["number"]).columns
    numerical_means = grouped[numerical_columns].mean()

    non_numerical_features = grouped.agg(
        {
            "keywords": combine_keywords,  # Handle keywords specifically
            "cluster_name": lambda x: x.mode()[0] if not x.mode().empty else "N/A",
        }
    )

    cluster_analysis = pd.concat([numerical_means, non_numerical_features], axis=1)
    cluster_analysis = cluster_analysis.drop(columns=["cluster"])
    cluster_analysis.to_csv(
        f"./output/analysis_{int(gower)}_{int(embeddings)}_{int(interact)}.csv"
    )
