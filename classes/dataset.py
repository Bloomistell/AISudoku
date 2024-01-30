from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class OneHotDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].to(torch.float)
        y = self.y[idx].to(torch.long)

        return X, y



class DenseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



class SimpleDynamicDataset(Dataset):
    def __init__(self, X, idx, y):
        '''
        Extracts dynamically the cell and its number from the data
        and creates the X input without this cell.

        Args:
        - X: shape (n, 81, 4), input data, with cells to remove, number
        are stored in binary to help the model make binary operations hopefully.
        - idx: shape (n, 50), list of possible cells to remove.
        - y: shape (n, 81), dense target data for cross-entropy loss.
        '''

        self.X = X
        self.idx = idx
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):

        idx = self.idx[i, 0].to(torch.long)
        y = self.y[i, idx].to(torch.long)

        X = self.X[i].to(torch.float)
        X[idx] = torch.zeros(1, 4, dtype=torch.float)

        return X, idx, y



class DynamicConvertDataset(Dataset):
    def __init__(self, X, y):
        '''
        Convert the binary data to floats dynamically.

        Args:
        - X: shape (n, 243), input data, with cells to remove, number
        are stored in binary to help the model make binary operations hopefully.
        - idx: shape (n, 50), list of possible cells to remove.
        - y: shape (n, 81), dense target data for cross-entropy loss.
        '''

        self.X = X
        self.idx = idx
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):

        idx = self.idx[i, 0].to(torch.long)
        y = self.y[i, idx].to(torch.long)

        X = self.X[i].to(torch.float)
        X[idx] = torch.zeros(1, 4, dtype=torch.float)

        return X, idx, y
