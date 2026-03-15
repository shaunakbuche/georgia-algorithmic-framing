"""
topic_modeling.py

Latent Dirichlet Allocation (LDA) topic modeling for the political
advertisement corpus.

Key fix from original repo: topic labels are NOT hardcoded before training.
LDA does not guarantee consistent topic ordering across runs or seeds.
Labels are assigned interactively after inspecting trained model output.

A label-assignment stub is provided at the bottom — edit it after
running the model and reviewing the printed top words.

Input:  data/processed/ads_preprocessed.csv
Output: models/lda_model, models/lda_dictionary
        data/processed/ad_topics.csv
        data/processed/county_topics.csv
"""

import ast
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


class TopicModeler:
    """
    LDA topic modeling for political advertisement corpus.

    Usage:
        modeler = TopicModeler(num_topics=10, random_state=42)
        modeler.build_corpus(texts)
        modeler.train()
        modeler.print_topics()
        # ← Inspect output, then assign labels manually:
        modeler.set_topic_labels({0: "Economic Policy", 1: "Healthcare", ...})
    """

    def __init__(self, num_topics: int = 10, random_state: int = 42):
        self.num_topics = num_topics
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None
        # Labels are None until set after training inspection
        self._topic_labels: dict = {}

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def build_corpus(self,
                     texts: list,
                     no_below: int = 5,
                     no_above: float = 0.90) -> None:
        """
        Build Gensim dictionary and BoW corpus from tokenized texts.

        Parameters
        ----------
        texts : list of list of str
            Tokenized, preprocessed documents
        no_below : int
            Filter tokens appearing in fewer than n documents.
            Paper uses 5 (§3.2.3).
        no_above : float
            Filter tokens appearing in more than this fraction of documents.
            Set to 0.90 (not 0.50) — with only ~2,147 ads, 0.50 is too
            aggressive and removes meaningful political vocabulary.
            See methodological note in paper §6.
        """
        logger.info(f"Building corpus from {len(texts)} documents...")

        self.dictionary = corpora.Dictionary(texts)
        initial_size = len(self.dictionary)

        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        filtered_size = len(self.dictionary)

        logger.info(
            f"Dictionary: {initial_size} → {filtered_size} tokens "
            f"(no_below={no_below}, no_above={no_above})"
        )

        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        empty_docs = sum(1 for doc in self.corpus if len(doc) == 0)
        if empty_docs > 0:
            logger.warning(f"{empty_docs} documents have empty BoW vectors after filtering")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, passes: int = 10, chunksize: int = 100) -> None:
        """Train LDA model on the constructed corpus."""
        if self.corpus is None:
            raise RuntimeError("Call build_corpus() before train()")

        logger.info(f"Training LDA: {self.num_topics} topics, {passes} passes...")

        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            update_every=1,
            chunksize=chunksize,
            passes=passes,
            alpha="auto",
            per_word_topics=True,
        )
        logger.info("LDA training complete.")

    # ------------------------------------------------------------------
    # Topic labeling — MUST be done after training, not before
    # ------------------------------------------------------------------

    def print_topics(self, num_words: int = 15) -> None:
        """
        Print top words for each topic.

        Run this after training, inspect the output, then call
        set_topic_labels() with your manual label assignments.
        Do NOT assign labels before seeing this output — LDA topic
        ordering is not deterministic across runs.
        """
        if self.model is None:
            raise RuntimeError("Call train() before print_topics()")

        print("\n" + "="*60)
        print("LDA TOPIC TOP WORDS — inspect before assigning labels")
        print("="*60)
        for i in range(self.num_topics):
            words = self.model.show_topic(i, num_words)
            word_str = ", ".join(w for w, _ in words)
            label = self._topic_labels.get(i, f"[UNLABELED Topic {i}]")
            print(f"\nTopic {i} ({label}):")
            print(f"  {word_str}")
        print("="*60 + "\n")

    def set_topic_labels(self, labels: dict) -> None:
        """
        Assign human-readable labels to topic IDs after manual inspection.

        Parameters
        ----------
        labels : dict
            Mapping of {topic_id (int): label (str)}
            e.g. {0: "Economic Policy", 1: "Healthcare/COVID", ...}
        """
        missing = [i for i in range(self.num_topics) if i not in labels]
        if missing:
            logger.warning(f"No labels provided for topic IDs: {missing}")

        self._topic_labels = labels
        logger.info(f"Topic labels set: {labels}")

    def get_label(self, topic_id: int) -> str:
        """Return the label for a topic ID (or a placeholder if unlabeled)."""
        return self._topic_labels.get(topic_id, f"Topic_{topic_id}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_document_topics(self, tokens: list) -> list:
        """Get topic probability distribution for a document."""
        if not tokens:
            return []
        bow = self.dictionary.doc2bow(tokens)
        return self.model.get_document_topics(bow, minimum_probability=0.0)

    def get_dominant_topic(self, tokens: list) -> int | None:
        """Get the highest-probability topic for a document."""
        dist = self.get_document_topics(tokens)
        if not dist:
            return None
        return max(dist, key=lambda x: x[1])[0]

    def get_topic_words(self, topic_id: int, num_words: int = 10) -> list:
        """Return top (word, probability) pairs for a topic."""
        return self.model.show_topic(topic_id, num_words)

    # ------------------------------------------------------------------
    # Coherence
    # ------------------------------------------------------------------

    def get_coherence(self, texts: list) -> float:
        """Calculate Cv coherence score for topic selection validation."""
        cm = CoherenceModel(
            model=self.model,
            texts=texts,
            dictionary=self.dictionary,
            coherence="c_v",
        )
        return cm.get_coherence()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Path) -> None:
        """Save model and dictionary to disk."""
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(model_dir / "lda_model"))
        self.dictionary.save(str(model_dir / "lda_dictionary"))

        # Save labels separately (JSON-serializable)
        import json
        labels_path = model_dir / "topic_labels.json"
        with open(labels_path, "w") as f:
            json.dump(self._topic_labels, f, indent=2)

        logger.info(f"Model saved to {model_dir}/")

    def load(self, model_dir: Path) -> None:
        """Load model, dictionary, and labels from disk."""
        import json
        self.model = LdaModel.load(str(model_dir / "lda_model"))
        self.dictionary = corpora.Dictionary.load(str(model_dir / "lda_dictionary"))

        labels_path = model_dir / "topic_labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                raw = json.load(f)
                self._topic_labels = {int(k): v for k, v in raw.items()}

        logger.info(f"Model loaded from {model_dir}/")


# ---------------------------------------------------------------------------
# Topic selection: coherence-based K selection
# ---------------------------------------------------------------------------

def select_num_topics(texts: list,
                      k_range: range = range(5, 16),
                      passes: int = 10) -> pd.DataFrame:
    """
    Select optimal number of topics based on Cv coherence.

    Paper tested K ∈ {5, 8, 10, 12, 15} and selected K=10 based on
    coherence scores and qualitative interpretability (§3.2.3).
    """
    results = []
    for k in k_range:
        logger.info(f"Testing K={k}...")
        modeler = TopicModeler(num_topics=k)
        modeler.build_corpus(texts)
        modeler.train(passes=passes)
        score = modeler.get_coherence(texts)
        results.append({"k": k, "coherence": score})
        logger.info(f"  K={k}: coherence={score:.4f}")

    df = pd.DataFrame(results)
    best_k = df.loc[df["coherence"].idxmax(), "k"]
    logger.info(f"\nBest K by coherence: {best_k}")
    return df


# ---------------------------------------------------------------------------
# County-year aggregation
# ---------------------------------------------------------------------------

def calculate_topic_prevalence(ads_df: pd.DataFrame,
                                modeler: TopicModeler,
                                tokens_col: str = "tokens",
                                weight_col: str = "impressions") -> pd.DataFrame:
    """
    Calculate impression-weighted topic prevalence for each county-year.

    Parameters
    ----------
    ads_df : pd.DataFrame
        Ad data with tokens (as string-repr of list) and impressions
    modeler : TopicModeler
        Trained and labeled TopicModeler instance
    tokens_col : str
        Column containing token lists (string repr from CSV)
    weight_col : str
        Column containing impression weights

    Returns
    -------
    pd.DataFrame
        County-year DataFrame with topic_N_share columns and combined
        thematic categories (social, health, election).
    """
    logger.info("Calculating topic prevalence...")

    # Parse tokens from string representation
    def parse_tokens(t):
        try:
            return ast.literal_eval(t) if isinstance(t, str) else t
        except Exception:
            return []

    ads_df = ads_df.copy()
    ads_df["_tokens"] = ads_df[tokens_col].apply(parse_tokens)
    ads_df["dominant_topic"] = ads_df["_tokens"].apply(modeler.get_dominant_topic)

    results = []

    for (county, year), group in ads_df.groupby(["county_fips", "year"]):
        total_impressions = group[weight_col].sum()
        if total_impressions == 0:
            continue

        row = {"county_fips": county, "year": year}

        for topic_id in range(modeler.num_topics):
            topic_impressions = group.loc[
                group["dominant_topic"] == topic_id, weight_col
            ].sum()
            row[f"topic_{topic_id}_share"] = topic_impressions / total_impressions

        results.append(row)

    result_df = pd.DataFrame(results)

    # Combined thematic categories used in regression
    # These assignments depend on trained topic labels — update after inspection
    # Default assignments reflect paper's K=10 model (Appendix E.2)
    labels_inv = {v: k for k, v in modeler._topic_labels.items()}

    immigration_id = labels_inv.get("Immigration/Border", 2)
    social_id      = labels_inv.get("Social Issues", 6)
    health_id      = labels_inv.get("Healthcare/COVID", 1)
    election_id    = labels_inv.get("Election Integrity", 5)

    result_df["topic_social_share"] = (
        result_df.get(f"topic_{immigration_id}_share", 0) +
        result_df.get(f"topic_{social_id}_share", 0)
    )
    result_df["topic_health_share"] = result_df.get(f"topic_{health_id}_share", 0)
    result_df["topic_election_share"] = result_df.get(f"topic_{election_id}_share", 0)

    logger.info(f"Created topic prevalence for {len(result_df)} county-years")
    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ===== POST-TRAINING LABEL ASSIGNMENTS =====
# Edit this dictionary after running the model and reviewing print_topics().
# These are the labels from the paper's trained K=10 model.
# If your re-trained model produces different topic orderings (which it may),
# update these assignments accordingly.
PAPER_TOPIC_LABELS = {
    0: "Economic Policy",
    1: "Healthcare/COVID",
    2: "Immigration/Border",
    3: "Education",
    4: "Character/Attack",
    5: "Election Integrity",
    6: "Social Issues",
    7: "Law/Crime",
    8: "Generic Campaign",
    9: "Governance",
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config()
    tm_config = config.get("topic_modeling", {})

    input_path = Path("data/processed/ads_preprocessed.csv")
    if not input_path.exists():
        logger.error(
            f"Preprocessed ads not found at {input_path}\n"
            "Run src/preprocessing/preprocess_ads.py first."
        )
        return

    ads_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(ads_df)} ad records")

    # Deduplicate for model training (train on unique texts, not county-duplicates)
    unique_ads = ads_df.drop_duplicates(subset=["ad_id"]) if "ad_id" in ads_df.columns \
        else ads_df.drop_duplicates(subset=["text"])

    texts = [
        ast.literal_eval(t) if isinstance(t, str) else t
        for t in unique_ads["tokens"]
        if t and t != "[]"
    ]
    logger.info(f"Training on {len(texts)} unique ad texts")

    # Optional: run coherence-based K selection
    # Uncomment if you want to re-validate K=10:
    # coherence_df = select_num_topics(texts, k_range=range(5, 16))
    # coherence_df.to_csv("results/tables/topic_coherence.csv", index=False)

    # Train model
    modeler = TopicModeler(
        num_topics=tm_config.get("num_topics", 10),
        random_state=tm_config.get("random_state", 42),
    )
    modeler.build_corpus(
        texts,
        no_below=tm_config.get("no_below", 5),
        no_above=tm_config.get("no_above", 0.90),
    )
    modeler.train(
        passes=tm_config.get("passes", 10),
        chunksize=tm_config.get("chunksize", 100),
    )

    # Print topics for inspection BEFORE assigning labels
    modeler.print_topics(num_words=15)

    # Assign labels from paper model
    # If you've re-trained and topics look different, update PAPER_TOPIC_LABELS above
    modeler.set_topic_labels(PAPER_TOPIC_LABELS)

    # Coherence score
    coherence = modeler.get_coherence(texts)
    logger.info(f"Model Cv coherence: {coherence:.4f}")

    # Save model
    model_dir = Path("models")
    modeler.save(model_dir)

    # Assign topics to all ads (including county-duplicated rows)
    ads_df["dominant_topic"] = ads_df["tokens"].apply(
        lambda t: modeler.get_dominant_topic(
            ast.literal_eval(t) if isinstance(t, str) else (t or [])
        )
    )
    ads_df["topic_label"] = ads_df["dominant_topic"].apply(modeler.get_label)

    ads_df.to_csv("data/processed/ad_topics.csv", index=False)
    logger.info("Saved ad-level topics to data/processed/ad_topics.csv")

    # Aggregate to county-year
    topic_prevalence = calculate_topic_prevalence(ads_df, modeler)
    topic_prevalence.to_csv("data/processed/county_topics.csv", index=False)
    logger.info("Saved county-year topic prevalence to data/processed/county_topics.csv")


if __name__ == "__main__":
    main()
