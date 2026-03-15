"""
preprocess_ads.py

Text preprocessing pipeline for political advertisement corpus.
Applies cleaning, tokenization, stopword removal, and lemmatization
to prepare ad text for both sentiment analysis and topic modeling.

Input:  data/processed/ads_attributed.csv
Output: data/processed/ads_preprocessed.csv
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data if not present
for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt"
                       else f"corpora/{resource}" if resource in ["stopwords", "wordnet"]
                       else f"taggers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# Domain-specific stopwords added to NLTK default list
POLITICAL_STOPWORDS = {
    "ad", "paid", "sponsored", "click", "learn", "more", "visit",
    "www", "com", "http", "https", "facebook", "instagram",
    "donate", "donation", "dollar", "fund", "contribute",
    "p.o.", "box", "authorized",  # Common ad disclaimer boilerplate
}


class TextPreprocessor:
    """
    Text preprocessing pipeline for political advertisement analysis.

    Two modes:
      - for_sentiment: Return cleaned string (preserve negations, punctuation
                       spacing) for TextBlob/VADER
      - for_lda:       Return token list (stopwords removed, lemmatized)
                       for Gensim LDA

    Using raw text for sentiment and preprocessed tokens for LDA is
    methodologically important — VADER/TextBlob need full sentences to
    handle negation ("not good"), while LDA needs bag-of-words tokens.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english")) | POLITICAL_STOPWORDS

    def clean_text(self, text: str) -> str:
        """Remove URLs, emails, special characters; normalize whitespace."""
        text = str(text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # Remove HTML entities
        text = re.sub(r"&[a-z]+;", " ", text)

        # Remove special characters, preserve apostrophes for contractions
        text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text.lower()

    def tokenize(self, text: str) -> list:
        """Tokenize text into words using NLTK punkt tokenizer."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """Remove stopwords and very short tokens."""
        return [t for t in tokens
                if t not in self.stop_words
                and len(t) > 2
                and not t.isdigit()]  # Remove pure numbers (keep years handled below)

    def preserve_years(self, tokens: list) -> list:
        """Re-add election-relevant years that digit filter would strip."""
        # Keep tokens that look like years in the study period
        relevant_years = {"2018", "2020", "2022", "2024", "2016", "2021"}
        return tokens  # Years are already kept since we filter isdigit() not all digits

    def lemmatize(self, tokens: list) -> list:
        """Lemmatize tokens to dictionary base form."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Minimal preprocessing for sentiment analysis.

        Preserves sentence structure, negations, and punctuation spacing
        that lexicon-based sentiment tools depend on.
        Only removes URLs and normalizes whitespace.
        """
        if pd.isna(text) or str(text).strip() == "":
            return ""

        text = str(text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_for_lda(self, text: str) -> list:
        """
        Full preprocessing pipeline for LDA topic modeling.

        Returns a list of cleaned, lemmatized tokens suitable for
        Gensim's bag-of-words conversion.
        """
        if pd.isna(text) or str(text).strip() == "":
            return []

        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        return tokens


def preprocess_ads_dataframe(ads_df: pd.DataFrame,
                              text_column: str = "text") -> pd.DataFrame:
    """
    Apply preprocessing pipeline to full ads DataFrame.

    Adds two columns:
      - text_clean:  minimal-cleaned text for sentiment analysis
      - tokens:      LDA-ready token list (stored as string for CSV serialization)

    Parameters
    ----------
    ads_df : pd.DataFrame
        DataFrame containing ad records with a text column
    text_column : str
        Name of the column containing raw ad text

    Returns
    -------
    pd.DataFrame
        Original DataFrame with preprocessing columns added
    """
    preprocessor = TextPreprocessor()
    df = ads_df.copy()

    logger.info(f"Preprocessing {len(df)} advertisements...")

    # Deduplicate on text before processing — same ad can appear multiple times
    # due to county attribution. Process unique texts, then merge back.
    unique_texts = df[[text_column]].drop_duplicates()

    logger.info(f"  Unique ad texts: {len(unique_texts)}")

    unique_texts["text_clean"] = unique_texts[text_column].apply(
        preprocessor.preprocess_for_sentiment
    )
    unique_texts["tokens"] = unique_texts[text_column].apply(
        preprocessor.preprocess_for_lda
    )

    # Store tokens as string representation for CSV (re-parse with ast.literal_eval)
    unique_texts["tokens_str"] = unique_texts["tokens"].apply(str)

    # Merge back to full DataFrame
    df = df.merge(
        unique_texts[[text_column, "text_clean", "tokens_str"]],
        on=text_column,
        how="left",
    )
    df = df.rename(columns={"tokens_str": "tokens"})

    # Log preprocessing summary
    empty_tokens = (df["tokens"] == "[]").sum()
    logger.info(f"  Empty token lists after preprocessing: {empty_tokens}")

    avg_tokens = df["tokens"].apply(lambda t: len(eval(t)) if t != "[]" else 0).mean()
    logger.info(f"  Average tokens per ad: {avg_tokens:.1f}")

    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path("data/processed/ads_attributed.csv")

    if not input_path.exists():
        logger.error(
            f"Attributed ads not found at {input_path}\n"
            "Run src/data/download_ad_data.py first."
        )
        return

    ads_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(ads_df)} attributed ad records")

    preprocessed = preprocess_ads_dataframe(ads_df)

    output_path = Path("data/processed/ads_preprocessed.csv")
    preprocessed.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed ads to {output_path}")


if __name__ == "__main__":
    main()
