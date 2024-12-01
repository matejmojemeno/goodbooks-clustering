import pandas as pd
from sentence_transformers import SentenceTransformer

print("Imported")

data = pd.read_csv("./goodbooks-10k/books_enriched.csv")
desc = data.description
desc = desc.dropna()
print(desc.info())
sentences = list(desc)

model = SentenceTransformer("all-MiniLM-L6-v2")


embeddings = model.encode(sentences)
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
