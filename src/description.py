import langid
import pandas as pd
import spacy


def process_description(text: str, nlp) -> str:
    """Processes a single description using spaCy NLP."""
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    words = [word for word in words if not any(ent.text == word for ent in doc.ents)]
    return " ".join(words)


def clean_description(description: pd.Series) -> pd.Series:
    """Cleans and preprocesses descriptions."""
    language = description.apply(lambda x: langid.classify(x)[0])
    description = description[language == "en"]  # Keep only English descriptions

    description = description.str.replace("'", "")
    description = description.str.replace("[^a-zA-Z]", " ", regex=True)
    description = description.str.replace("\s+", " ", regex=True).str.strip()
    description = description.str.lower()

    nlp = spacy.load("en_core_web_sm")
    description = description.apply(lambda x: process_description(x, nlp))
    return description


def get_cleaned_description(books: pd.DataFrame) -> pd.DataFrame:
    """Loads the Goodreads dataset and cleans the descriptions."""
    books: pd.DataFrame = pd.read_csv("./goodbooks-10k/books_enriched.csv")
    books["description_clean"] = None

    description: pd.Series = books.description.dropna()
    description = clean_description(description)

    books.loc[description.index, "description_clean"] = description
    books = books.dropna(subset=["description_clean"]).reset_index(drop=True)
    return books
