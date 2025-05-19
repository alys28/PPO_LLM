import torch
from torch.utils.data import Dataset, DataLoader
import json
# Import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tokenizer import Tokenizer


class MathDataset(Dataset):
    def __init__(self, data_file, device, vocab, max_answer_len=15):
        self.data = []
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.max_answer_len = max_answer_len
        self.tokenizer = Tokenizer(vocab, max_answer_len - 1)
        self.device = device
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_embedding = torch.tensor(item['embedding'], dtype=torch.float, device=self.device)  # Move to device
        answer_text = str(item['answer'])  # Convert answer to string (e.g., "-543808")
        decoder_input = self.tokenizer.encode(answer_text, add_start_token=True, add_end_token=False, add_pad_token=True)
        output = self.tokenizer.encode(answer_text, add_start_token=False, add_end_token=True, add_pad_token=True)
        decoder_input = torch.tensor(decoder_input, dtype=torch.long, device=self.device)
        output = torch.tensor(output, dtype=torch.long, device=self.device)
        key_padding_mask = [True] + (decoder_input != self.tokenizer.pad_token_id).tolist() # Add True to the beginning of the list to account for the question embedding
        key_padding_mask = torch.tensor(key_padding_mask, dtype=torch.long)
        key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = torch.tril(torch.ones(self.max_answer_len, self.max_answer_len), diagonal=0).int()
        causal_mask = causal_mask.unsqueeze(0)
        # broadcast bitwise AND operation
        causal_mask = causal_mask.to(self.device)
        key_padding_mask = key_padding_mask.to(self.device)
        return input_embedding, decoder_input, output, causal_mask, key_padding_mask 

def get_math_dataloader(data_file, device, vocab, batch_size, max_answer_len, shuffle=True):
    dataset = MathDataset(data_file, device, vocab, max_answer_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_math_val_dataloader(data_file, device, vocab, batch_size, max_answer_len, shuffle=False):
    dataset = MathDataset(data_file, device, vocab, max_answer_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)