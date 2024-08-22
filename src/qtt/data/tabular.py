import torch
from torch.utils.data import Dataset

# class TabularCurveDataset:
#     def __init__(
#         self,
#         processed_X: pd.DataFrame,
#         curve: pd.DataFrame,
#     ):
#         super().__init__()

#         self.processed_X = processed_X
#         self.curve = curve


class TabularCurveDataset(Dataset):
    def __init__(self, X, curve, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.curve = torch.tensor(curve, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)


    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.curve[idx], self.y[idx]
