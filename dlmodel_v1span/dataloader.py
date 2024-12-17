import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Step 1: Define the custom dataset class


class SDDataset(Dataset):
    def __init__(self, csv_file,):
        # Load the data into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        # Separate features and target
        self.score = self.data['composite_score'].values
        self.features = self.data.drop(columns=['composite_score']).values
        self.targets = self.data['composite_score'].values

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the features and target for the sample at index idx
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return features, target, torch.tensor(self.score[idx], dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, df,):
        # Load the data into a pandas DataFrame
        self.data = df
        # Separate features and target
        self.features = self.data.values

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the features and target for the sample at index idx
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        return features

# Step 2: Initialize the dataset and DataLoader


def create_traindataloader(data_file, batch_size=4, shuffle=True):
    dataset = SDDataset(data_file,)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_testdataloader(data_file, batch_size=1):
    dataset = TestDataset(data_file,)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# train_loader = create_traindataloader(
#     f"/STORAGE/peter/PREPARE/dlmodel/train_2021.csv", batch_size=4)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data.size())
#     print(target)
