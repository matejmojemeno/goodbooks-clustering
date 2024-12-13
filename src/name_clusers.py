import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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

    books, cluster_keywords = get_keywords(books)
    cluster_names = {}

    for cluster, keywords in cluster_keywords.items():
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a literary analyst tasked with identifying the broad theme or genre of a book collection. "
                    "Your response should be as concise as possible, using only 5-10 words to describe the collection. "
                    "Avoid detailed explanations, specific examples, or explicit mentions of people, places, or niche details. "
                    "Instead, focus on a broad, high-level genre or theme that connects the keywords provided. "
                    "\n\nGuidelines:\n"
                    "- If keywords suggest unrelated topics, summarize them as 'Diverse collection' or 'Mixed themes.'\n"
                    "- For keywords mentioning specific individuals or places, generalize to their associated genre (e.g., 'Historical nonfiction,' 'Fantasy').\n"
                    "- Avoid repeating phrases like 'Based on the keywords' or 'The theme appears to be.'\n"
                    "- Keep the response neutral and generalized without adding unnecessary interpretations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are examples:\n"
                    "Keywords: magic, school, wizard, adventure -> Fantasy books about magical adventures.\n"
                    "Keywords: detective, murder, mystery, investigation -> Mystery books about solving crimes.\n"
                    "Keywords: travel, cooking, memoir, photography -> Nonfiction about experiences and hobbies.\n"
                    "Keywords: space, galaxy, aliens, technology -> Science fiction about space exploration.\n"
                    "Now, identify the theme for this collection of books based on these keywords: "
                    f"{', '.join(keywords)}"
                ),
            },
        ]

        output = generator(messages, max_new_tokens=256)
        cluster_names[cluster] = output[0]["generated_text"][-1]["content"]

    books["cluster_name"] = books["cluster"].map(cluster_names)
    return books
