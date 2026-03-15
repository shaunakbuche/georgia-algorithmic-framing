"""
tests/test_topic_modeling.py

Unit tests for the TopicModeler class.
Focus: label-assignment after training, corpus construction, inference.

Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.topic_modeling import TopicModeler


SAMPLE_TEXTS = [
    ["job", "economy", "tax", "business", "growth", "wage"],
    ["health", "care", "covid", "pandemic", "vaccine", "hospital"],
    ["border", "immigration", "illegal", "security", "enforcement"],
    ["school", "child", "education", "teacher", "student", "learn"],
    ["corrupt", "radical", "extreme", "lie", "attack", "failed"],
    ["vote", "election", "fraud", "ballot", "count", "integrity"],
    ["gun", "abortion", "right", "family", "freedom", "faith"],
    ["police", "crime", "safety", "defund", "law", "order"],
    ["georgia", "support", "help", "win", "together", "join"],
    ["government", "washington", "leader", "fight", "people", "change"],
    ["job", "economy", "growth", "worker", "small", "business"],
    ["health", "insurance", "coverage", "protect", "doctor", "care"],
]


class TestTopicModelerInit:
    def test_defaults(self):
        m = TopicModeler()
        assert m.num_topics == 10
        assert m.random_state == 42

    def test_custom_params(self):
        m = TopicModeler(num_topics=5, random_state=7)
        assert m.num_topics == 5
        assert m.random_state == 7

    def test_no_labels_initially(self):
        m = TopicModeler()
        assert m._topic_labels == {}

    def test_not_trained_initially(self):
        m = TopicModeler()
        assert m.model is None
        assert m.dictionary is None


class TestCorpusBuilding:
    def test_corpus_created(self):
        m = TopicModeler(num_topics=3)
        m.build_corpus(SAMPLE_TEXTS, no_below=1, no_above=0.99)
        assert m.corpus is not None
        assert len(m.corpus) == len(SAMPLE_TEXTS)

    def test_dictionary_created(self):
        m = TopicModeler(num_topics=3)
        m.build_corpus(SAMPLE_TEXTS, no_below=1, no_above=0.99)
        assert m.dictionary is not None
        assert len(m.dictionary) > 0

    def test_no_below_filtering(self):
        m1 = TopicModeler(num_topics=3)
        m1.build_corpus(SAMPLE_TEXTS, no_below=1, no_above=0.99)

        m2 = TopicModeler(num_topics=3)
        m2.build_corpus(SAMPLE_TEXTS, no_below=10, no_above=0.99)

        # Stricter no_below → smaller dictionary
        assert len(m1.dictionary) >= len(m2.dictionary)

    def test_empty_texts_handled(self):
        texts_with_empty = SAMPLE_TEXTS + [[]]
        m = TopicModeler(num_topics=3)
        m.build_corpus(texts_with_empty, no_below=1, no_above=0.99)
        # Should not raise; empty doc creates empty BoW vector
        assert len(m.corpus) == len(texts_with_empty)


class TestTraining:
    @pytest.fixture
    def trained_modeler(self):
        m = TopicModeler(num_topics=3, random_state=42)
        m.build_corpus(SAMPLE_TEXTS, no_below=1, no_above=0.99)
        m.train(passes=2)
        return m

    def test_model_created(self, trained_modeler):
        assert trained_modeler.model is not None

    def test_correct_num_topics(self, trained_modeler):
        assert trained_modeler.model.num_topics == 3

    def test_get_topic_words(self, trained_modeler):
        words = trained_modeler.get_topic_words(0, num_words=5)
        assert len(words) == 5
        assert all(isinstance(w, str) for w, _ in words)
        assert all(isinstance(p, float) for _, p in words)

    def test_dominant_topic_valid(self, trained_modeler):
        tokens = ["economy", "job", "tax"]
        topic_id = trained_modeler.get_dominant_topic(tokens)
        assert topic_id is None or (0 <= topic_id < 3)

    def test_empty_tokens_returns_none(self, trained_modeler):
        assert trained_modeler.get_dominant_topic([]) is None

    def test_document_topics_sum_to_one(self, trained_modeler):
        tokens = ["health", "vaccine", "pandemic"]
        dist = trained_modeler.get_document_topics(tokens)
        total = sum(p for _, p in dist)
        assert abs(total - 1.0) < 0.01

    def test_train_requires_corpus(self):
        m = TopicModeler(num_topics=3)
        with pytest.raises(RuntimeError, match="build_corpus"):
            m.train()

    def test_print_topics_requires_model(self, capsys):
        m = TopicModeler(num_topics=3)
        with pytest.raises(RuntimeError, match="train"):
            m.print_topics()


class TestTopicLabels:
    """
    Critical tests: labels must be assigned AFTER training,
    not hardcoded before. This was the original repo's key bug.
    """

    @pytest.fixture
    def trained_modeler(self):
        m = TopicModeler(num_topics=3, random_state=42)
        m.build_corpus(SAMPLE_TEXTS, no_below=1, no_above=0.99)
        m.train(passes=2)
        return m

    def test_labels_empty_before_assignment(self, trained_modeler):
        assert trained_modeler._topic_labels == {}, \
            "Labels must be empty until set_topic_labels() is called"

    def test_set_labels(self, trained_modeler):
        labels = {0: "Economic", 1: "Healthcare", 2: "Immigration"}
        trained_modeler.set_topic_labels(labels)
        assert trained_modeler._topic_labels == labels

    def test_get_label_after_set(self, trained_modeler):
        trained_modeler.set_topic_labels({0: "Economy", 1: "Health", 2: "Border"})
        assert trained_modeler.get_label(0) == "Economy"
        assert trained_modeler.get_label(1) == "Health"

    def test_get_label_fallback(self, trained_modeler):
        # Unlabeled topics return a placeholder, not crash
        assert trained_modeler.get_label(99) == "Topic_99"

    def test_partial_labels_warning(self, trained_modeler, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            trained_modeler.set_topic_labels({0: "Economy"})  # Missing 1 and 2
        assert "No labels provided" in caplog.text

    def test_labels_survive_inference(self, trained_modeler):
        """Label assignments should not affect inference results."""
        tokens = ["job", "economy"]
        topic_before = trained_modeler.get_dominant_topic(tokens)

        trained_modeler.set_topic_labels({0: "Economy", 1: "Health", 2: "Border"})
        topic_after = trained_modeler.get_dominant_topic(tokens)

        assert topic_before == topic_after, \
            "Label assignment must not change inference results"
