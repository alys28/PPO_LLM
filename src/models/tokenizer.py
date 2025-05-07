class Tokenizer:
    def __init__(self, vocab, max_seq_len=16):
        self.vocab = vocab
        self.unk_token = "<unk>"
        self.pad_token = "<pad>" # Used for padding sequences to the same length during batch training
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.vocab_size = len(vocab) + 4  # +4 for special tokens
        self.unk_token_id = vocab.get(self.unk_token, 0)
        self.pad_token_id = vocab.get(self.pad_token, 1)
        self.start_token_id = vocab.get(self.start_token, 2)
        self.end_token_id = vocab.get(self.end_token, 3)
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.max_seq_len = max_seq_len
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs."""
        tokens = text.split()
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        if add_special_tokens:
            token_ids = [self.start_token_id] + token_ids + [self.end_token_id]
        return token_ids
    def encode_for_batch(self, text):
        """Convert text to token IDs for training."""
        token_ids = self.encode(text)
        padded_token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))
        return padded_token_ids
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(token_id, self.unk_token) for token_id in token_ids]
        return " ".join(tokens)
    def pad_sequence(self, sequences, max_length=None):
        """Pad sequences to the same length."""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
        return padded_sequences