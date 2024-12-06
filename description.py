import langid
import pandas as pd
import spacy


def process_description(text: str, nlp) -> str:
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop)


def clean_description(description: pd.Series) -> pd.Series:
    language = description.apply(lambda x: langid.classify(x)[0])
    description = pd.Series(description[language == "en"])

    description = description.str.replace("'", "")
    description = description.str.replace("[^a-zA-Z]", " ", regex=True)
    description = description.str.replace("\s+", " ", regex=True).str.strip()
    description = description.str.lower()

    nlp = spacy.load("en_core_web_sm")

    def preprocess(text: str) -> str:
        doc = nlp(text)
        return " ".join(token.lemma_ for token in doc if not token.is_stop)

    description = pd.Series(description.apply(preprocess))
    return description


def main():
    books: pd.DataFrame = pd.read_csv("./goodbooks-10k/books_enriched.csv")
    books["description_clean"] = None
    description: pd.Series = books.description.dropna()

    print(description.shape)

    print(description.loc[0])
    description = clean_description(description)
    print(description.loc[0])

    print(description.shape)

    books.loc[description.index, "description_clean"] = description
    books.to_csv("./goodbooks-10k/books_enriched.csv", index=False)


if __name__ == "__main__":
    main()
