"""
Topic Modeling Module

This module implements Latent Dirichlet Allocation (LDA) topic modeling
for political advertisement analysis.
"""

import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

# Topic labels based on manual interpretation
TOPIC_LABELS = {
    0: 'Economic Policy',
    1: 'Healthcare/COVID',
    2: 'Immigration/Border',
    3: 'Education',
    4: 'Character/Attack',
    5: 'Election Integrity',
    6: 'Social Issues',
    7: 'Law/Crime',
    8: 'Generic Campaign',
    9: 'Governance'
}


class TopicModeler:
    """
    LDA topic modeling for political advertisements.
    
    Attributes
    ----------
    num_topics : int
        Number of topics to extract
    random_state : int
        Random seed for reproducibility
    dictionary : gensim.corpora.Dictionary
        Word-to-id mapping
    corpus : list
        Bag-of-words corpus
    model : gensim.models.LdaModel
        Trained LDA model
    """
    
    def __init__(self, num_topics: int = 10, random_state: int = 42):
        self.num_topics = num_topics
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None
    
    def build_corpus(self, texts: list, 
                     no_below: int = 5, 
                     no_above: float = 0.5):
        """
        Build dictionary and corpus from tokenized texts.
        
        Parameters
        ----------
        texts : list of list
            Tokenized and preprocessed texts
        no_below : int
            Filter terms appearing in fewer than n documents
        no_above : float
            Filter terms appearing in more than fraction of documents
        """
        logger.info(f"Building corpus from {len(texts)} documents...")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(texts)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        logger.info(f"Dictionary size: {len(self.dictionary)}")
        logger.info(f"Corpus size: {len(self.corpus)}")
    
    def train(self, passes: int = 10, chunksize: int = 100):
        """
        Train LDA model.
        
        Parameters
        ----------
        passes : int
            Number of passes through corpus
        chunksize : int
            Number of documents in each training chunk
        """
        logger.info(f"Training LDA model with {self.num_topics} topics...")
        
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            update_every=1,
            chunksize=chunksize,
            passes=passes,
            alpha='auto',
            per_word_topics=True
        )
        
        logger.info("LDA training complete.")
    
    def get_coherence(self, texts: list) -> float:
        """
        Calculate Cv coherence score.
        
        Parameters
        ----------
        texts : list of list
            Original tokenized texts
        
        Returns
        -------
        float
            Coherence score
        """
        coherence_model = CoherenceModel(
            model=self.model,
            texts=texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    
    def get_document_topics(self, tokens: list) -> list:
        """
        Get topic distribution for a document.
        
        Parameters
        ----------
        tokens : list
            Tokenized document
        
        Returns
        -------
        list
            List of (topic_id, probability) tuples
        """
        bow = self.dictionary.doc2bow(tokens)
        return self.model.get_document_topics(bow)
    
    def get_dominant_topic(self, tokens: list) -> int:
        """
        Get the dominant topic for a document.
        
        Parameters
        ----------
        tokens : list
            Tokenized document
        
        Returns
        -------
        int
            Dominant topic ID
        """
        topic_dist = self.get_document_topics(tokens)
        if not topic_dist:
            return None
        return max(topic_dist, key=lambda x: x[1])[0]
    
    def get_topic_words(self, topic_id: int, num_words: int = 10) -> list:
        """
        Get top words for a topic.
        
        Parameters
        ----------
        topic_id : int
            Topic ID
        num_words : int
            Number of top words to return
        
        Returns
        -------
        list
            List of (word, probability) tuples
        """
        return self.model.show_topic(topic_id, num_words)
    
    def print_topics(self, num_words: int = 10):
        """Print all topics with their top words."""
        for i in range(self.num_topics):
            label = TOPIC_LABELS.get(i, f'Topic {i}')
            words = self.get_topic_words(i, num_words)
            word_str = ', '.join([w for w, _ in words])
            print(f"\nTopic {i} ({label}):")
            print(f"  {word_str}")
    
    def save(self, model_path: Path, dict_path: Path):
        """Save model and dictionary to disk."""
        self.model.save(str(model_path))
        self.dictionary.save(str(dict_path))
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Dictionary saved to {dict_path}")
    
    def load(self, model_path: Path, dict_path: Path):
        """Load model and dictionary from disk."""
        self.model = LdaModel.load(str(model_path))
        self.dictionary = corpora.Dictionary.load(str(dict_path))
        logger.info(f"Model loaded from {model_path}")


def select_num_topics(texts: list, k_range: range = range(5, 16)) -> pd.DataFrame:
    """
    Select optimal number of topics based on coherence.
    
    Parameters
    ----------
    texts : list of list
        Tokenized texts
    k_range : range
        Range of topic numbers to try
    
    Returns
    -------
    pd.DataFrame
        Results with k and coherence scores
    """
    results = []
    
    for k in k_range:
        logger.info(f"Testing K={k}...")
        modeler = TopicModeler(num_topics=k)
        modeler.build_corpus(texts)
        modeler.train()
        coherence = modeler.get_coherence(texts)
        results.append({'k': k, 'coherence': coherence})
        logger.info(f"K={k}: Coherence={coherence:.4f}")
    
    return pd.DataFrame(results)


def calculate_topic_prevalence(ads_df: pd.DataFrame,
                                modeler: TopicModeler,
                                tokens_col: str = 'tokens',
                                weight_col: str = 'impressions') -> pd.DataFrame:
    """
    Calculate impression-weighted topic prevalence for county-years.
    
    Parameters
    ----------
    ads_df : pd.DataFrame
        Advertisement data with tokens and impressions
    modeler : TopicModeler
        Trained topic modeler
    tokens_col : str
        Column containing tokenized text
    weight_col : str
        Column containing impression weights
    
    Returns
    -------
    pd.DataFrame
        County-year topic prevalence shares
    """
    logger.info("Calculating topic prevalence...")
    
    # Get dominant topic for each ad
    ads_df['dominant_topic'] = ads_df[tokens_col].apply(modeler.get_dominant_topic)
    
    results = []
    
    for (county, year), group in ads_df.groupby(['county_fips', 'year']):
        total_impressions = group[weight_col].sum()
        
        row = {'county_fips': county, 'year': year}
        
        for topic_id in range(modeler.num_topics):
            topic_impressions = group[
                group['dominant_topic'] == topic_id
            ][weight_col].sum()
            
            share = topic_impressions / total_impressions if total_impressions > 0 else 0
            row[f'topic_{topic_id}_share'] = share
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Add combined topic categories
    result_df['topic_social_share'] = (
        result_df['topic_2_share'] +  # Immigration
        result_df['topic_6_share']    # Social Issues
    )
    result_df['topic_health_share'] = result_df['topic_1_share']
    result_df['topic_election_share'] = result_df['topic_5_share']
    
    logger.info(f"Created topic prevalence for {len(result_df)} county-years")
    
    return result_df


def main():
    """Main execution function."""
    # Load preprocessed advertisement data
    ads_path = Path('data/processed/ads_preprocessed.csv')
    
    if not ads_path.exists():
        logger.error(f"Preprocessed ads not found at {ads_path}")
        return
    
    logger.info(f"Loading data from {ads_path}")
    ads_df = pd.read_csv(ads_path)
    
    # Convert tokens column from string to list
    import ast
    ads_df['tokens'] = ads_df['tokens'].apply(ast.literal_eval)
    
    # Get all token lists
    texts = ads_df['tokens'].tolist()
    
    # Train model
    modeler = TopicModeler(num_topics=10, random_state=42)
    modeler.build_corpus(texts)
    modeler.train(passes=10)
    
    # Calculate coherence
    coherence = modeler.get_coherence(texts)
    logger.info(f"Model coherence: {coherence:.4f}")
    
    # Print topics
    modeler.print_topics()
    
    # Save model
    modeler.save(
        Path('models/lda_model.pkl'),
        Path('models/dictionary.pkl')
    )
    
    # Calculate topic prevalence
    topic_prevalence = calculate_topic_prevalence(ads_df, modeler)
    
    # Save topic assignments
    ads_df['dominant_topic'] = ads_df['tokens'].apply(modeler.get_dominant_topic)
    ads_df['topic_label'] = ads_df['dominant_topic'].map(TOPIC_LABELS)
    ads_df.to_csv('data/processed/ad_topics.csv', index=False)
    
    # Save county-year topic prevalence
    topic_prevalence.to_csv('data/processed/county_topics.csv', index=False)
    
    logger.info("Topic modeling complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
