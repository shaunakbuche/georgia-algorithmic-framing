#!/usr/bin/env python3
"""
Topic Modeling for Political Advertisements

This script performs LDA topic modeling on political advertisement text
to identify prevalent themes and frames in campaign messaging.

Usage:
    python topic_modeling.py [--num-topics K]
    
Options:
    --num-topics K    Number of topics (default: 10)
"""

import argparse
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pickle
import os
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RANDOM_STATE = 42


class TopicModeler:
    """
    LDA topic modeling for political advertisement analysis.
    """
    
    def __init__(self, num_topics=10, random_state=42):
        self.num_topics = num_topics
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.topic_labels = None
    
    def prepare_corpus(self, texts):
        """
        Prepare corpus for LDA training.
        
        Parameters
        ----------
        texts : list of list of str
            Tokenized documents (list of token lists)
        """
        print("Preparing corpus...")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(texts)
        print(f"  Initial vocabulary size: {len(self.dictionary)}")
        
        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=5,      # Remove terms in <5 docs
            no_above=0.5,    # Remove terms in >50% of docs
            keep_n=5000      # Keep top 5000 terms
        )
        print(f"  Filtered vocabulary size: {len(self.dictionary)}")
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        print(f"  Corpus size: {len(self.corpus)} documents")
    
    def train(self, texts=None, passes=10, chunksize=100):
        """
        Train LDA model.
        
        Parameters
        ----------
        texts : list of list of str, optional
            Tokenized documents (if not already prepared)
        passes : int
            Number of passes through corpus
        chunksize : int
            Number of documents per training chunk
        """
        if texts is not None:
            self.prepare_corpus(texts)
        
        print(f"\nTraining LDA model with {self.num_topics} topics...")
        
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            update_every=1,
            chunksize=chunksize,
            passes=passes,
            alpha='auto',
            eta='auto',
            per_word_topics=True
        )
        
        print("  Training complete!")
    
    def get_coherence_score(self, texts):
        """
        Calculate coherence score for model.
        
        Parameters
        ----------
        texts : list of list of str
            Tokenized documents
            
        Returns
        -------
        float
            Cv coherence score
        """
        coherence_model = CoherenceModel(
            model=self.model,
            texts=texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    
    def get_topic_words(self, topic_id, top_n=15):
        """
        Get top words for a topic.
        
        Parameters
        ----------
        topic_id : int
            Topic index
        top_n : int
            Number of top words to return
            
        Returns
        -------
        list of tuple
            (word, probability) pairs
        """
        return self.model.show_topic(topic_id, top_n)
    
    def get_all_topic_words(self, top_n=15):
        """
        Get top words for all topics.
        
        Returns
        -------
        dict
            Topic ID -> list of (word, probability) pairs
        """
        return {
            i: self.get_topic_words(i, top_n)
            for i in range(self.num_topics)
        }
    
    def get_document_topics(self, doc_bow):
        """
        Get topic distribution for a document.
        
        Parameters
        ----------
        doc_bow : list of tuple
            Bag-of-words representation
            
        Returns
        -------
        list of tuple
            (topic_id, probability) pairs
        """
        return self.model.get_document_topics(doc_bow, minimum_probability=0.0)
    
    def get_primary_topic(self, doc_bow):
        """
        Get most probable topic for a document.
        
        Parameters
        ----------
        doc_bow : list of tuple
            Bag-of-words representation
            
        Returns
        -------
        int
            Topic ID with highest probability
        """
        topics = self.get_document_topics(doc_bow)
        if topics:
            return max(topics, key=lambda x: x[1])[0]
        return None
    
    def set_topic_labels(self, labels):
        """
        Set human-readable labels for topics.
        
        Parameters
        ----------
        labels : dict
            Topic ID -> label mapping
        """
        self.topic_labels = labels
    
    def save_model(self, filepath):
        """Save model and dictionary to file."""
        self.model.save(filepath)
        self.dictionary.save(filepath + '.dict')
        if self.topic_labels:
            with open(filepath + '.labels', 'wb') as f:
                pickle.dump(self.topic_labels, f)
    
    def load_model(self, filepath):
        """Load model and dictionary from file."""
        self.model = LdaModel.load(filepath)
        self.dictionary = corpora.Dictionary.load(filepath + '.dict')
        if os.path.exists(filepath + '.labels'):
            with open(filepath + '.labels', 'rb') as f:
                self.topic_labels = pickle.load(f)


def select_optimal_k(texts, k_range=(5, 16)):
    """
    Select optimal number of topics using coherence scores.
    
    Parameters
    ----------
    texts : list of list of str
        Tokenized documents
    k_range : tuple
        (min_k, max_k) range to search
        
    Returns
    -------
    pd.DataFrame
        K values and coherence scores
    """
    print("\nSearching for optimal number of topics...")
    results = []
    
    for k in range(k_range[0], k_range[1]):
        print(f"\n  Testing K={k}...")
        modeler = TopicModeler(num_topics=k)
        modeler.train(texts, passes=5)  # Fewer passes for speed
        coherence = modeler.get_coherence_score(texts)
        
        results.append({
            'k': k,
            'coherence': coherence
        })
        print(f"    Coherence: {coherence:.4f}")
    
    df = pd.DataFrame(results)
    best_k = df.loc[df['coherence'].idxmax(), 'k']
    print(f"\nOptimal K: {best_k} (coherence: {df['coherence'].max():.4f})")
    
    return df


def assign_document_topics(modeler, texts, df):
    """
    Assign topics to all documents.
    
    Parameters
    ----------
    modeler : TopicModeler
        Trained topic modeler
    texts : list of list of str
        Tokenized documents
    df : pd.DataFrame
        Original dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with topic columns added
    """
    print("\nAssigning topics to documents...")
    
    df = df.copy()
    
    # Initialize topic probability columns
    for i in range(modeler.num_topics):
        df[f'topic_{i}_prob'] = 0.0
    
    primary_topics = []
    
    for idx, text in enumerate(tqdm(texts, desc="Assigning topics")):
        bow = modeler.dictionary.doc2bow(text)
        topics = modeler.get_document_topics(bow)
        
        for topic_id, prob in topics:
            df.loc[idx, f'topic_{topic_id}_prob'] = prob
        
        primary_topics.append(modeler.get_primary_topic(bow))
    
    df['primary_topic'] = primary_topics
    
    # Add topic labels if available
    if modeler.topic_labels:
        df['primary_topic_label'] = df['primary_topic'].map(modeler.topic_labels)
    
    return df


def aggregate_topics_county_year(df, modeler):
    """
    Aggregate topic distributions to county-year level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Ad-level data with topic assignments
    modeler : TopicModeler
        Trained topic modeler (for labels)
        
    Returns
    -------
    pd.DataFrame
        County-year topic distributions
    """
    print("\nAggregating topics to county-year level...")
    
    topic_cols = [f'topic_{i}_prob' for i in range(modeler.num_topics)]
    
    def weighted_topic_means(group):
        weights = group['impressions']
        result = {}
        
        for col in topic_cols:
            result[col.replace('_prob', '_share')] = np.average(
                group[col], weights=weights
            )
        
        # Primary topic distribution (by impressions)
        topic_impressions = group.groupby('primary_topic')['impressions'].sum()
        total_impr = group['impressions'].sum()
        
        for topic_id in range(modeler.num_topics):
            key = f'topic_{topic_id}_primary_share'
            if topic_id in topic_impressions.index:
                result[key] = topic_impressions[topic_id] / total_impr
            else:
                result[key] = 0.0
        
        return pd.Series(result)
    
    aggregated = df.groupby(['county', 'year']).apply(weighted_topic_means).reset_index()
    
    return aggregated


def print_topic_summary(modeler):
    """Print summary of discovered topics."""
    print("\n" + "="*60)
    print("TOPIC SUMMARY")
    print("="*60)
    
    for topic_id in range(modeler.num_topics):
        words = modeler.get_topic_words(topic_id, top_n=10)
        word_str = ', '.join([w for w, p in words])
        
        label = ""
        if modeler.topic_labels and topic_id in modeler.topic_labels:
            label = f" ({modeler.topic_labels[topic_id]})"
        
        print(f"\nTopic {topic_id}{label}:")
        print(f"  {word_str}")


# Default topic labels (adjust based on actual model output)
DEFAULT_LABELS = {
    0: "Economic Policy",
    1: "Healthcare/COVID",
    2: "Immigration/Border",
    3: "Education",
    4: "Character/Attack",
    5: "Election Integrity",
    6: "Social Issues",
    7: "Law/Crime",
    8: "Generic Campaign",
    9: "Governance"
}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Topic modeling for political ads')
    parser.add_argument('--num-topics', type=int, default=10,
                       help='Number of topics (default: 10)')
    parser.add_argument('--search-k', action='store_true',
                       help='Search for optimal K')
    args = parser.parse_args()
    
    print("="*60)
    print("TOPIC MODELING")
    print("="*60)
    
    # Load preprocessed data
    print("\nLoading preprocessed advertisement data...")
    ad_filepath = os.path.join(DATA_DIR, 'processed', 'ads_preprocessed.csv')
    df = pd.read_csv(ad_filepath)
    print(f"Loaded {len(df):,} advertisements")
    
    # Convert token strings to lists
    texts = [str(t).split() for t in df['tokens']]
    texts = [t for t in texts if len(t) >= 3]  # Filter very short docs
    print(f"Using {len(texts):,} documents with 3+ tokens")
    
    # Optionally search for optimal K
    if args.search_k:
        k_results = select_optimal_k(texts, k_range=(5, 16))
        k_results.to_csv(
            os.path.join(RESULTS_DIR, 'topic_k_search.csv'),
            index=False
        )
    
    # Train final model
    modeler = TopicModeler(num_topics=args.num_topics, random_state=RANDOM_STATE)
    modeler.train(texts, passes=10)
    
    # Calculate coherence
    coherence = modeler.get_coherence_score(texts)
    print(f"\nFinal model coherence (Cv): {coherence:.4f}")
    
    # Set topic labels
    modeler.set_topic_labels(DEFAULT_LABELS)
    
    # Print topic summary
    print_topic_summary(modeler)
    
    # Assign topics to documents
    df_topics = assign_document_topics(modeler, texts, df.head(len(texts)))
    
    # Aggregate to county-year
    county_year_topics = aggregate_topics_county_year(df_topics, modeler)
    
    # Save outputs
    print("\nSaving outputs...")
    os.makedirs(os.path.join(RESULTS_DIR, 'model_outputs'), exist_ok=True)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, 'model_outputs', 'lda_model')
    modeler.save_model(model_path)
    print(f"  Saved model to {model_path}")
    
    # Save ad-level topics
    ad_output = os.path.join(DATA_DIR, 'processed', 'ad_topics.csv')
    df_topics.to_csv(ad_output, index=False)
    print(f"  Saved ad-level topics to {ad_output}")
    
    # Save county-year topics
    cy_output = os.path.join(DATA_DIR, 'processed', 'county_year_topics.csv')
    county_year_topics.to_csv(cy_output, index=False)
    print(f"  Saved county-year topics to {cy_output}")
    
    # Save coherence score
    with open(os.path.join(RESULTS_DIR, 'topic_coherence.txt'), 'w') as f:
        f.write(f"Number of topics: {args.num_topics}\n")
        f.write(f"Cv coherence score: {coherence:.4f}\n")
    
    print("\n" + "="*60)
    print("TOPIC MODELING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
