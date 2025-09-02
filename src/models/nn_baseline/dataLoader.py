import torch
from torch.utils.data import Dataset, DataLoader
import json
from src.models.nn_baseline.math_scaler import MathScaler

class MathDataset(Dataset):

    def __init__(self, data_file, device, normalizer=None, fit_normalizer=False):
        self.data = []
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.X = torch.tensor([entry["embedding"] for entry in self.data], device=device, dtype=torch.float32)
        self.Y = torch.tensor([entry["answer"] for entry in self.data], device=device, dtype=torch.float32).unsqueeze(1)
        
        # Apply scaling if scaler is provided
        if normalizer is not None:
            if fit_normalizer:
                # Fit the scaler on this dataset
                normalizer.fit(self.Y.cpu())
            # Apply scaling
            self.Y = normalizer.transform(self.Y)
        else:
            # Fallback to simple normalization (for backward compatibility)
            mean = self.Y.mean()
            std = self.Y.std()
            self.Y = (self.Y - mean) / std
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
       return self.X[idx], self.Y[idx]
    
def get_math_dataloader(data_file, device, batch_size, shuffle=True, normalizer=None, fit_normalizer=False):
    dataset = MathDataset(data_file, device, normalizer=normalizer, fit_normalizer=fit_normalizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_math_val_dataloader(data_file, device, batch_size, shuffle=False, normalizer=None, fit_normalizer=False):
    dataset = MathDataset(data_file, device, normalizer=normalizer, fit_normalizer=fit_normalizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)