import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
# Step 1: Define the custom dataset class
bins_2016 = [0, 17, 21, 22, 24, 32, 38, 45, 52, 56, 62, 66, 70, 74, 77, 80, 84, 87, 90, 92, 93, 94, 96, 97, 99, 101, 103, 104, 105, 107, 109, 111, 113, 115, 116, 118, 119, 121, 122, 124, 127, 128, 129, 131, 133, 134, 135, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 155, 156, 158, 159,
             160, 161, 163, 164, 165, 166, 168, 170, 172, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 186, 187, 189, 190, 191, 194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 206, 208, 210, 212, 213, 216, 217, 219, 221, 223, 225, 227, 229, 231, 234, 235, 237, 241, 244, 246, 249, 251, 253, 256, 260, 264, 267, 272, 278, 285, 296, 384]
labels_2016 = [13, 19, 22, 23, 29, 35, 42, 48, 54, 60, 64, 68, 72, 76, 79, 83, 86, 89, 91, 93, 94, 95, 97, 98, 100, 102, 104, 105, 106, 108, 110, 112, 114, 116, 117, 119, 120, 122, 123, 126, 128, 129, 130, 132, 134, 135, 136, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 154, 156, 157, 159, 160, 161,
               162, 164, 165, 166, 167, 169, 171, 173, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 190, 191, 193, 195, 196, 197, 199, 200, 201, 202, 203, 204, 205, 207, 209, 211, 213, 215, 217, 218, 220, 222, 224, 226, 228, 230, 232, 235, 236, 239, 243, 245, 248, 250, 252, 255, 258, 263, 266, 270, 275, 282, 290, 315]
bins_2021 = [0, 20, 24, 27, 32, 38, 41, 49, 52, 57, 59, 63, 66, 70, 71, 72, 73, 76, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
             158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 229, 230, 231, 232, 233, 234, 235, 237, 239, 241, 243, 244, 247, 250, 251, 253, 255, 258, 263, 267, 271, 278, 287, 295, 309, 384]
labels_2021 = [16, 22, 26, 29, 34, 40, 45, 51, 55, 58, 61, 65, 68, 71, 72, 73, 75, 77, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
               158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 226, 228, 230, 231, 232, 233, 234, 235, 236, 238, 240, 242, 244, 246, 249, 251, 252, 254, 257, 260, 265, 269, 274, 283, 291, 303, 321]


class SDDataset(Dataset):
    def __init__(self, csv_file, year):
        # Load the data into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        if year == 2016:
            bins = bins_2016
            labels = labels_2016
        if year == 2021:
            bins = bins_2021
            labels = labels_2021
        # bins, self.labels = self.cluster_score(self.data)
        # Separate features and target
        self.score = self.data['composite_score'].values
        self.features = self.data.drop(columns=['composite_score']).values
        self.targets = pd.cut(
            self.data['composite_score'], bins=bins, labels=labels).astype(int).values

    def cluster_score(self, df):
        key = Counter(df['composite_score'].values)
        k = sorted(key.keys())
        c = key[k[0]]
        value = key[k[0]]*k[0]
        span = [k[0]]
        v = []
        for i in k[1:]:
            c += key[i]
            if c+key[i] < 15:
                value += key[i]*i
                if i == k[-1]:
                    span.append(i)
                    v.append(int(value/c))
                    value = 0
            else:
                span.append(i)
                value += key[i]*i
                v.append(int(value/c))
                value = 0
                c = 0
        return span, v

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the features and target for the sample at index idx
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return features, target, torch.tensor(self.score[idx], dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, df, year):
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


def create_traindataloader(data_file, year, batch_size=4, shuffle=True):
    dataset = SDDataset(data_file, year)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_testdataloader(data_file, year, batch_size=1):
    dataset = TestDataset(data_file, year)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# train_loader = create_traindataloader(
#     f"/STORAGE/peter/PREPARE/dlmodel/train_2021.csv", batch_size=4)
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data.size())
#     print(target)
