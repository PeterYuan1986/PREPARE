import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
# Step 1: Define the custom dataset class
bins_2016 = [0,
             21,
             29,
             45,
             56,
             66,
             74,
             80,
             87,
             91,
             94,
             98,
             103,
             107,
             111,
             115,
             119,
             123,
             128,
             131,
             134,
             138,
             140,
             143,
             146,
             148,
             151,
             155,
             158,
             161,
             163,
             165,
             168,
             172,
             175,
             177,
             179,
             181,
             183,
             186,
             189,
             191,
             195,
             198,
             200,
             203,
             207,
             211,
             213,
             217,
             222,
             225,
             229,
             234,
             238,
             246,
             251,
             256,
             264,
             272,
             287,
             384]
labels_2016 = [0,
               24,
               37,
               51,
               62,
               70,
               77,
               84,
               89,
               93,
               96,
               101,
               105,
               109,
               113,
               117,
               121,
               126,
               129,
               133,
               136,
               139,
               142,
               144,
               147,
               149,
               153,
               157,
               159,
               162,
               164,
               166,
               170,
               174,
               176,
               178,
               180,
               182,
               185,
               188,
               190,
               193,
               196,
               199,
               202,
               205,
               209,
               212,
               215,
               219,
               224,
               227,
               231,
               236,
               243,
               248,
               253,
               260,
               268,
               280,
               384]
bins_2021 = [0,
             24,
             32,
             41,
             54,
             59,
             66,
             71,
             75,
             78,
             81,
             83,
             86,
             88,
             89,
             91,
             92,
             96,
             97,
             100,
             102,
             103,
             104,
             105,
             107,
             108,
             110,
             112,
             114,
             115,
             117,
             119,
             121,
             122,
             124,
             125,
             127,
             129,
             130,
             131,
             133,
             135,
             136,
             137,
             138,
             140,
             141,
             142,
             143,
             144,
             145,
             146,
             147,
             149,
             150,
             151,
             152,
             154,
             155,
             156,
             157,
             158,
             159,
             160,
             162,
             163,
             164,
             165,
             167,
             169,
             170,
             171,
             172,
             173,
             175,
             176,
             178,
             179,
             180,
             181,
             182,
             183,
             184,
             185,
             186,
             187,
             189,
             191,
             193,
             194,
             195,
             198,
             200,
             202,
             205,
             208,
             210,
             212,
             213,
             215,
             217,
             218,
             220,
             222,
             225,
             229,
             231,
             234,
             237,
             241,
             246,
             250,
             254,
             261,
             269,
             286,
             309,
             384]
labels_2021 = [19,
               27,
               37,
               49,
               57,
               63,
               69,
               73,
               77,
               79,
               82,
               85,
               87,
               89,
               90,
               92,
               94,
               97,
               98,
               101,
               103,
               104,
               105,
               106,
               108,
               109,
               111,
               113,
               115,
               116,
               118,
               120,
               122,
               123,
               125,
               126,
               128,
               130,
               131,
               132,
               134,
               136,
               137,
               138,
               139,
               141,
               142,
               143,
               144,
               145,
               146,
               147,
               148,
               150,
               151,
               152,
               153,
               155,
               156,
               157,
               158,
               159,
               160,
               161,
               163,
               164,
               165,
               166,
               168,
               170,
               171,
               172,
               173,
               174,
               176,
               177,
               179,
               180,
               181,
               182,
               183,
               184,
               185,
               186,
               187,
               188,
               190,
               192,
               194,
               195,
               197,
               199,
               201,
               204,
               207,
               209,
               211,
               213,
               214,
               216,
               218,
               219,
               221,
               224,
               227,
               230,
               233,
               236,
               239,
               243,
               248,
               252,
               257,
               266,
               276,
               296,
               321]


class SDDataset(Dataset):
    def __init__(self, csv_file, year):
        # Load the data into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        if year == 2016:
            bins = bins_2016
            self.labels = labels_2016
        if year == 2021:
            bins = bins_2021
            self.labels = labels_2021
        # bins, self.labels = self.cluster_score(self.data)
        # Separate features and target
        self.score = self.data['composite_score'].values
        self.features = self.data.drop(columns=['composite_score']).values
        self.targets = [self.labels.index(x) for x in pd.cut(
            self.data['composite_score'], bins=bins, labels=self.labels).astype(int).values]

    def cluster_score(self, df):
        key = Counter(df['composite_score'].values)
        k = sorted(key.keys())
        c = key[k[0]]
        value = key[k[0]]*k[0]
        span = [k[0]]
        v = []
        for i in k[1:]:
            c += key[i]
            if c+key[i] < 30:
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
