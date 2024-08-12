import re
import numpy as np
import nltk
from collections import defaultdict
from math import log2, inf

class HindiTokenizerEvaluation:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def split_hindi_words(self, text):
        # Regular expression for splitting Hindi words
        pattern = re.compile(r'\b[\u0900-\u097F]+\b')
        words = pattern.findall(text)
        return words

    def fertility_score(self, text):
        tokens = self.tokenizer.encode(text)
        words = self.split_hindi_words(text)
        num_words = len(words)
        num_tokens = len(tokens)
        fertility_score = num_tokens / num_words if num_words > 0 else 0
        return fertility_score

    def vocabulary_size(self):
        return len(self.tokenizer.vocab)

    def token_length_distribution(self, text):
        tokens = self.tokenizer.encode(text)
        token_lengths = [len(str(token)) for token in tokens]
        return np.histogram(token_lengths, bins=range(max(token_lengths) + 2))

    def token_coverage(self, text):
        tokens = self.tokenizer.encode(text)
        words = self.split_hindi_words(text)
        covered_words = sum(1 for word in words if word in tokens)
        return covered_words / len(words) if words else 0

    def subword_count(self, text):
        tokens = self.tokenizer.encode(text)
        words = self.split_hindi_words(text)
        subwords_per_word = len(tokens) / len(words) if words else 0
        return subwords_per_word

    def compression_ratio(self, text):
        tokens = self.tokenizer.encode(text)
        original_length = len(text)
        tokenized_length = sum(len(str(token)) for token in tokens)
        compression_ratio = tokenized_length / original_length if original_length > 0 else inf
        return compression_ratio

    def perplexity(self, text):
        tokens = self.tokenizer.encode(text)
        n = len(tokens)
        if n == 0:
            return inf
        log_prob_sum = sum(log2(1.0 / self.token_probability(token)) for token in tokens)
        perplexity = 2 ** (log_prob_sum / n)
        return perplexity

    def token_probability(self, token):
        # Simplified assumption: uniform distribution over vocabulary
        vocab_size = self.vocabulary_size()
        return 1.0 / vocab_size if vocab_size > 0 else 0

    def consistency(self, texts):
        token_sets = [set(self.tokenizer.encode(text)) for text in texts]
        common_tokens = set.intersection(*token_sets) if token_sets else set()
        consistency_ratio = len(common_tokens) / len(set.union(*token_sets)) if token_sets else 0
        return consistency_ratio

    def robustness_to_oov(self, text, oov_words):
        tokens = self.tokenizer.encode(text)
        oov_handling = sum(1 for word in oov_words if word in tokens)
        return oov_handling / len(oov_words) if oov_words else 0

    def tokenization_speed(self, text, repetitions=100):
        import time
        start_time = time.time()
        for _ in range(repetitions):
            self.tokenizer.encode(text)
        end_time = time.time()
        return (end_time - start_time) / repetitions

    def alignment_with_linguistic_units(self, text):
        tokens = self.tokenizer.encode(text)
        words = self.split_hindi_words(text)
        aligned_tokens = sum(1 for word in words if word in tokens)
        return aligned_tokens / len(words) if words else 0

    def entropy(self, text):
        tokens = self.tokenizer.encode(text)
        token_freqs = defaultdict(int)
        for token in tokens:
            token_freqs[token] += 1
        probs = [freq / len(tokens) for freq in token_freqs.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
        return entropy

    def bleu_score_comparison(self, reference_tokenizer, text):
        from nltk.translate.bleu_score import sentence_bleu
        reference_tokens = reference_tokenizer.encode(text)
        candidate_tokens = self.tokenizer.encode(text)
        score = sentence_bleu([reference_tokens], candidate_tokens)
        return score

    def character_coverage(self, text):
        chars = set(text)
        covered_chars = sum(1 for char in chars if char in self.tokenizer.vocab)
        return covered_chars / len(chars) if chars else 0

    def back_translation_score(self, translated_text, original_text):
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, translated_text, original_text).ratio()
        return ratio

