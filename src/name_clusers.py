import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .genres import get_genres


def get_keywords(books, max_df=0.5):
    grouped = books.groupby("cluster")["description_words"].apply(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=max_df, ngram_range=(1, 2)
    )
    tf_idf_matrix = vectorizer.fit_transform(grouped)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = {}

    for i, row in enumerate(tf_idf_matrix):
        cluster = grouped.index[i]
        feature_index = row.nonzero()[1]
        tfidf_scores = zip(feature_index, [row[0, x] for x in feature_index])
        cluster_keywords[cluster] = [
            feature_names[i]
            for i, s in sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:10]
        ]

    books["keywords"] = books["cluster"].map(cluster_keywords)
    return books, cluster_keywords


def get_info(books):
    books = books.drop(["title", "description_clean", "description_words"], axis=1)
    clusters = books.groupby("cluster").mean()

    genres = get_genres(books, 10)

    for col in clusters.columns:
        if col not in genres:
            print(col)
            clusters[col] = clusters[col].apply(
                lambda x: sorted(list(clusters[col])).index(x) + 1
            )

    cluster_info = []
    for i, row in clusters.iterrows():
        cluster_info.append(
            " | ".join([f"{col}: {row[col]}" for col in sorted(clusters.columns)])
        )

    return cluster_info


def name_clusters(books):
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    cluster_info = get_info(books)
    books, cluster_keywords = get_keywords(books)
    cluster_names = {}

    for cluster, keywords in cluster_keywords.items():
        metadata_info = cluster_info[cluster]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a literary analyst tasked with naming a cluster of books using both keywords and numeric metadata. "
                    "Genres are represented as continuous features ranging from 0 to 1, indicating the percentage of books in the cluster that belong to that genre. "
                    "Other metadata features range from 1 to 20, where 1 represents the lowest value and 20 the highest value for that feature across all clusters. "
                    "Values close to 1 or 20 for non-genre features may also be significant, providing context for terms like 'minimalist,' 'short,' or 'modern.' "
                    "Your response must be a descriptive, creative, and concise name for the cluster (6-10 words), summarizing its dominant themes or genres.\n\n"
                    "### Guidelines:\n"
                    "- Focus on **themes, genres, and narratives** inferred from keywords and metadata.\n"
                    "- Avoid starting responses with template phrases like:\n"
                    "  * 'Based on the keywords and metadata...'\n"
                    "  * 'The theme for this cluster appears to be...'\n"
                    "Instead, provide the cluster name directly.\n"
                    "- Avoid using the provided keywords verbatim in the cluster name unless they are extremely common (e.g., 'love,' 'magic,' 'war,' 'family'). Instead:\n"
                    "  * Interpret the keywords to convey their broader thematic or emotional resonance.\n"
                    "  * Use synonyms, analogies, or thematic summaries to reflect the cluster’s essence.\n"
                    "- When a keyword is common and unavoidable, integrate it seamlessly into a more creative or general context. For example:\n"
                    "  * 'Love' -> 'Timeless Tales of Romantic Endeavors.'\n"
                    "  * 'War' -> 'Epic Chronicles of Conflict and Survival.'\n"
                    "- **Do not use specific names, job titles, or characters** in the cluster name. Generalize them into broader concepts.\n"
                    "- Strive for creative and varied language; avoid repetitive structures like 'X and Y of Z.'\n"
                    "- Highlight genres with high percentages (e.g., 0.8 or above) as dominant themes.\n"
                    "- Interpret metadata features creatively:\n"
                    "  * Values near **20**: Use terms like 'modern,' 'detailed,' 'lengthy,' or 'complex.'\n"
                    "  * Values near **1**: Use terms like 'minimalist,' 'short,' or 'concise.'\n"
                    "- For clusters with diverse or conflicting themes, use names like 'Eclectic Mix of Diverse Narratives.'\n"
                    "- Avoid direct replication of keywords unless unavoidable."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are examples of ideal cluster names:\n"
                    "Keywords: magic, school, wizard, adventure; Metadata: fantasy: 0.85, mystery: 0.05, romance: 0.1; original_publication_year: 5 -> Fantasy Adventures in Magical Worlds of the Past\n"
                    "Keywords: detective, murder, investigation, crime; Metadata: mystery: 0.9, historical-fiction: 0.2, nonfiction: 0.05; pages: 15 -> Intricate Murder Mysteries with Forensic Depth\n"
                    "Keywords: travel, cooking, memoir, photography; Metadata: nonfiction: 0.95, general-fiction: 0.05; contemporary: 18 -> Modern Nonfiction About Creative Hobbies and Experiences\n"
                    "Keywords: unrelated, diverse, mixed, varied; Metadata: no dominant genre; original_publication_year: 10 -> Eclectic Mix of Diverse Narratives\n"
                    "Keywords: haiku, poetry, brief; Metadata: pages: 1, nonfiction: 0.9 -> Minimalist Poetry Exploring Human Experience\n\n"
                    "Now, identify the theme for this cluster based on:\n"
                    f"Keywords: {', '.join(keywords)}\n"
                    f"Metadata: {metadata_info}"
                ),
            },
        ]

        output = generator(messages, max_new_tokens=25, return_full_text=False)
        output = output[0]["generated_text"].split("\n")[0].strip()
        # Keep the first line of the response and truncate to avoid verbosity
        print("--------------------------------")
        print(output)
        print(len(output.split()))
        print("--------------------------------")
        cluster_names[cluster] = output

    books["cluster_name"] = books["cluster"].map(cluster_names)
    return books
