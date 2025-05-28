import torch
from torch.utils.data import Dataset, DataLoader
import json

class MathDataset(Dataset):

    def __init__(self, data_file, device):
        self.data = []
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.X = torch.tensor([entry["embedding"] for entry in self.data], device=device, dtype=torch.float32)
        self.Y = torch.tensor([entry["answer"] for entry in self.data], device=device, dtype=torch.float32).unsqueeze(1)
        # Normalize
        mean = self.Y.mean()
        std = self.Y.std()
        self.Y = (self.Y - mean) / std
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
       return self.X[idx], self.Y[idx]
    
def get_math_dataloader(data_file, device, batch_size, shuffle=True):
    dataset = MathDataset(data_file, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_math_val_dataloader(data_file, device, batch_size, shuffle=False):
    dataset = MathDataset(data_file, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)