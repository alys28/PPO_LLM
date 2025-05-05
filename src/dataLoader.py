import torch
from torch.utils.data import Dataset, DataLoader
import json

class MathDataset(Dataset):
    def __init__(self, data_file, pad_token_id=12, max_answer_len=10):
        self.data = []
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.pad_token_id = pad_token_id
        self.max_answer_len = max_answer_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_embedding = torch.tensor(item['embedding'], dtype=torch.float)  # Precomputed embedding
        answer_text = str(item['answer'])  # Convert answer to string (e.g., "-543808")
        # Tokenize the answer text (tokens: [0, 1, 2, ... 9, 10, 11, 12] where 0-9 are digits and 10 is the minus token, 11 is the end token, 12 is the pad token)
        token_ids = []
        for char in answer_text:
            if char == '-':
                token_ids.append(10)
            elif char == ' ':
                token_ids.append(11)
            else:
                token_ids.append(int(char))
        # Pad or truncate the tokenized answer sequence
        if len(token_ids) < self.max_answer_len:
            token_ids += [self.pad_token_id] * (self.max_answer_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_answer_len]

        target = torch.tensor(token_ids, dtype=torch.long)

        return input_embedding, target

def get_math_dataloader(data_file, batch_size=32, shuffle=True):
    dataset = MathDataset(data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_math_val_dataloader(data_file, batch_size=32, shuffle=False):
    dataset = MathDataset(data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)