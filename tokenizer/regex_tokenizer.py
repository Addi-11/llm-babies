from base_tokenizer import Tokenizer
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)


    def train(self, text, vocab_size=276, verbose=False):
        assert vocab_size >= self.default_vocab_size, "vocab_size must be greater than or equal to default_vocab_size"
        num_merges = vocab_size - self.default_vocab_size

        # create utf-8 encoded tokens from the text
        text_chunks = re.findall(self.compiled_pattern, text) # this returns into list of splits words
        ids = [list(text.encode("utf-8")) for text in text_chunks] # list of list of tokens

        self.vocab = self._build_vocab()

        for i in range(num_merges):
            stats = {}
            for chunk_id in ids:
                stats = self.get_stats(chunk_id, stats)
            top_pair = max(stats, key=stats.get)
            idx = self.default_vocab_size + i

            # update ids
            ids = [self.merge(chunk_id, top_pair, idx) for chunk_id in ids]

            # update merges and data
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({self.vocab[idx]} had {stats[top_pair]} occurences)")


    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    def _encode_chunks(self, chunk_bytes):
        """return chunk ids"""
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)

        return ids

    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunks(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
