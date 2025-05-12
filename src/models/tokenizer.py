import torch

class Tokenizer:
    def __init__(self, vocab, max_seq_len=16):
        self.vocab = vocab
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        # self.sep_token = "<sep>" # no need to add sep token since the embeddings are not part of the answer
        # Create token mappings
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Add special tokens
        special_tokens = [self.unk_token, self.pad_token, self.start_token, self.end_token]
        for token in special_tokens:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.vocab_size = len(self.token_to_id)
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.start_token_id = self.token_to_id[self.start_token]
        self.end_token_id = self.token_to_id[self.end_token]
        # self.sep_token_id = self.token_to_id[self.sep_token]
        self.max_seq_len = max_seq_len
    def __len__(self):
        """Return the size of the vocabulary."""
        return self.vocab_size

    def get_token_id(self, token):
        """Convert a token to its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            raise ValueError(f"Token {token} not found in vocabulary")

    def get_token(self, token_id):
        """Convert a token ID to its token."""
        return self.id_to_token.get(token_id, self.unk_token)

    def encode(self, text, add_start_token=False, add_end_token=True, add_pad_token=True):
        """Convert text to token IDs."""
        assert len(text) <= self.max_seq_len, "text length {} must be less than or equal to max_seq_len {}".format(len(text), self.max_seq_len)
        token_ids = [self.get_token_id(token) for token in text]
        if add_start_token:
            token_ids = [self.start_token_id] + token_ids
        if add_end_token:
            token_ids = token_ids + [self.end_token_id]
        if add_pad_token:
            token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))
        return token_ids[:self.max_seq_len]  # Truncate to max_seq_len

    def encode_for_batch(self, text, add_start_token=False, add_end_token=True, add_pad_token=True):
        """Convert text to token IDs for training."""
        token_ids = self.encode(text, add_start_token, add_end_token, add_pad_token)
        padded_token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))
        return padded_token_ids

    def decode(self, token_ids):
        """Convert token IDs back to text."""
        # If token_ids is a tensor, convert it to a list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = [self.get_token(token_id) for token_id in token_ids]
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

        