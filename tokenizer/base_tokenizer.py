class Tokenizer:
    """Base class for tokenizer"""
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.default_vocab_size = 256
    
    def get_stats(self, tokens, counts=None):
        """Get the count for each pair"""
        if counts is None:
            counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        """replace pair -> idx in ids"""
        i = 0
        new_ids = []
        while i < len(ids):
            if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids
    
    def train(self, text, vocab_size=276):
        """training tokenizer on vocabulary of vocab_size from text"""
        raise NotImplementedError
    
    def encode(self, text):
        """encoding logic"""
        raise NotImplementedError
    
    def decode(self, ids):
        """decoding logic"""
        raise NotImplementedError
    
    def _build_vocab(self):
        """Deriving vocab from the merges"""
        vocab = {idx: bytes([idx]) for idx in range(self.default_vocab_size)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab