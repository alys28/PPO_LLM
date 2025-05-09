import torch
from torch.utils.data import Dataset, DataLoader
import json
# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tokenizer import Tokenizer


class MathDataset(Dataset):
    def __init__(self, data_file, vocab, max_answer_len=15):
        self.data = []
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.max_answer_len = max_answer_len
        self.tokenizer = Tokenizer(vocab)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_embedding = torch.tensor(item['embedding'], dtype=torch.float)  # Precomputed embedding
        answer_text = str(item['answer'])  # Convert answer to string (e.g., "-543808")
        token_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        decoder_input = [self.tokenizer.sep_token_id] + token_ids
        output = token_ids + [self.tokenizer.end_token_id]
        if len(decoder_input) < self.max_answer_len:
            token_ids += [self.tokenizer.pad_token_id] * (self.max_answer_len - len(token_ids))
        elif len(decoder_input) > self.max_answer_len:
            raise ValueError(f"Answer length exceeds max_answer_len: {len(token_ids)} > {self.max_answer_len}")
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        output = torch.tensor(output, dtype=torch.long)
        # Create a mask such that the padded tokens are ignored
        decoder_input = decoder_input.unsqueeze(0)
        output = output.unsqueeze(0)
        decoder_input = decoder_input.to(input_embedding.device)
        output = output.to(input_embedding.device)
        # Create a mask for the decoder input
        padding_mask = (decoder_input != self.tokenizer.pad_token_id).unsqueeze(0)
        decoder_mask = padding_mask & torch.tril(torch.ones(self.max_answer_len, self.max_answer_len), diagonal=1).int()
        decoder_mask = decoder_mask.to(decoder_input.device)
        return input_embedding, decoder_input, output, decoder_mask 

def get_math_dataloader(data_file, vocab, batch_size=32, shuffle=True):
    dataset = MathDataset(data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_math_val_dataloader(data_file, vocab, batch_size=32, shuffle=False):
    dataset = MathDataset(data_file, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)