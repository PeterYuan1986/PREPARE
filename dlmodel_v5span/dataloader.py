import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Step 1: Define the custom dataset class
bins_2016 = list(range(0, 385, 5))
labels_2016 = [4.0, 9.333333333333334, 13.666666666666666, 18.076923076923077, 22.583333333333332, 28.0, 32.72727272727273, 37.77777777777778, 43.44444444444444, 47.54545454545455, 53.38461538461539, 57.90909090909091, 62.94117647058823, 67.95, 73.25, 78.04347826086956, 83.55555555555556, 88.1923076923077, 92.78787878787878, 97.82352941176471, 103.28571428571429, 108.28125, 113.03225806451613, 117.97297297297297, 122.83333333333333, 128.44444444444446, 133.28205128205127, 138.42857142857142, 143.37777777777777, 147.92727272727274, 152.95238095238096, 158.17073170731706, 163.13725490196077, 167.725,
               173.5744680851064, 177.67391304347825, 182.72549019607843, 187.80434782608697, 192.93023255813952, 198.04444444444445, 203.02083333333334, 208.09677419354838, 212.87878787878788, 217.8684210526316, 223.25, 228.06666666666666, 232.8181818181818, 237.6315789473684, 242.89473684210526, 247.64285714285714, 252.66666666666666, 257.95652173913044, 263.3529411764706, 267.9375, 272.3076923076923, 278.0, 283.375, 287.8888888888889, 292.6666666666667, 296.6666666666667, 303.5, 307.3333333333333, 312.5, 317.6666666666667, 325.0, 330.0, 333.5, 337.5, 342.5, 347.5, 352.5, 357.5, 362.5, 367.5, 372.5, 377.5]
bins_2021 = list(range(0, 385, 5))
labels_2021 = [2.5, 7.5, 13.333333333333334, 18.714285714285715, 23.38888888888889, 27.6, 32.75, 38.416666666666664, 42.2, 48.07692307692308, 53.0, 58.04761904761905, 63.1764705882353, 68.03846153846153, 72.46153846153847, 77.95238095238095, 82.94, 88.04411764705883, 92.58, 97.91525423728814, 103.22368421052632, 108.22222222222223, 113.3013698630137, 118.12, 122.84782608695652, 128.025, 133.1038961038961, 137.6705882352941, 143.21505376344086, 148.07777777777778, 153.03333333333333, 158.02247191011236, 163.09574468085106, 168.26315789473685,
               172.78125, 178.1012658227848, 182.95294117647057, 187.71830985915494, 193.32558139534885, 198.12698412698413, 202.95, 208.20833333333334, 213.13636363636363, 218.04545454545453, 223.06521739130434, 228.12121212121212, 233.0612244897959, 237.925, 242.85185185185185, 248.0, 252.9090909090909, 258.11764705882354, 262.85714285714283, 267.6666666666667, 272.5833333333333, 278.125, 283.2, 287.6, 293.5, 298.5, 303.0, 307.8333333333333, 311.5, 317.5, 322.5, 327.5, 331.5, 337.5, 342.5, 347.5, 352.5, 357.5, 362.5, 367.5, 372.5, 377.5]


class SDDataset(Dataset):
    def __init__(self, csv_file, year):
        self.data = pd.read_csv(csv_file)
        if year == 2016:
            bins = bins_2016
            self.labels = labels_2016
        if year == 2021:
            bins = bins_2021
            self.labels = labels_2021
        self.length = len(self.labels)
        # Load the data into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        # Separate features and target
        self.score = self.data['composite_score'].values
        self.features = self.data.drop(columns=['composite_score']).values
        self.targets = [self.labels.index(x) for x in pd.cut(
            self.data['composite_score'], bins=bins, labels=self.labels).values]

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the features and target for the sample at index idx
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return features, target, torch.tensor(self.score[idx], dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, df):
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
    dataset = SDDataset(data_file, year,)
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
