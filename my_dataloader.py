import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, window_size, csv_path,column):
        df = pd.read_csv(csv_path)
        ys = np.array(df[column])
        self.scale = max(ys)
        self.ys = ys / self.scale

        self.window_size = window_size

    def __len__(self):
        return len(self.ys) - self.window_size

    def __getitem__(self, idx):
        data_x = self.ys[idx:idx + self.window_size].squeeze()
        data_y = self.ys[idx + self.window_size]

        return torch.tensor(data_x).to(torch.float), torch.tensor(data_y).to(torch.float)
    def get_scale(self):
        return self.scale