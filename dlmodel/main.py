import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from network import Net, TransformerBinaryClassifier
from dataloader import create_traindataloader, labels_2021, labels_2016
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import numpy as np
import json
from datetime import datetime
hyperparameters = {
    # fold 0
    'ce_2021_0': 0.1,
    'mse_2021_0': 0.01,
    'l1_2021_0': 1,
    'lr_2021_0': 0.0001,
    'weight_decay_2021_0': 0.01,
    # fold 1
    'ce_2021_1': 1,
    'mse_2021_1': 1,
    'l1_2021_1': 100,
    'lr_2021_1': 0.001,
    'weight_decay_2021_1': 0.001,
    # fold 2
    'ce_2021_2': 0.5,
    'mse_2021_2': 0.01,
    'l1_2021_2': 1,
    'lr_2021_2': 0.00001,
    'weight_decay_2021_2': 0.001,
    # fold 3
    'ce_2021_3': 0.5,
    'mse_2021_3': 0.01,
    'l1_2021_3': 1,
    'lr_2021_3': 0.00001,
    'weight_decay_2021_3': 0.001,
    # fold 4
    'ce_2021_4': 0.5,
    'mse_2021_4': 0.01,
    'l1_2021_4': 1,
    'lr_2021_4': 0.00001,
    'weight_decay_2021_4': 0.001,
    'ce_2016_0': 0.5,
    'mse_2016_0': 0.01,
    'l1_2016_0': 1,
    'lr_2016_0': 0.00001,
    'weight_decay_2016_0': 0.001,
    'ce_2016_1': 0.5,
    'mse_2016_1': 0.01,
    'l1_2016_1': 1,
    'lr_2016_1': 0.00001,
    'weight_decay_2016_1': 0.001,
    'ce_2016_2': 0.5,
    'mse_2016_2': 0.01,
    'l1_2016_2': 1,
    'lr_2016_2': 0.00001,
    'weight_decay_2016_2': 0.001,
    'ce_2016_3': 0.5,
    'mse_2016_3': 0.01,
    'l1_2016_3': 1,
    'lr_2016_3': 0.00001,
    'weight_decay_2016_3': 0.001,
    'ce_2016_4': 0.5,
    'mse_2016_4': 0.01,
    'l1_2016_4': 1,
    'lr_2016_4': 0.00001,
    'weight_decay_2016_4': 0.001,
}

best_rmse = float('inf')


def log_message(message, args, file_name="log.txt", ):
    print(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name.replace('.txt', f'_{args.year}_{args.fold}.txt'), "a") as log_file:
        log_file.write(f"{timestamp} - {message}\n")


def train(args, model, device, train_loader, lossfunction, mseloss, l1loss, optimizer, epoch):
    model.train()
    test_CE_loss = 0
    test_MSE_loss = 0
    test_l1_loss = 0
    rmse = 0
    ce = hyperparameters[f"ce_{args.year}_{args.fold}"]
    mse = hyperparameters[f"mse_{args.year}_{args.fold}"]
    l1 = hyperparameters[f"l1_{args.year}_{args.fold}"]
    if args.year == 2016:
        labels = labels_2016
    elif args.year == 2021:
        labels = labels_2021

    for batch_idx, (data, target, score) in enumerate(train_loader):
        data, target = data.to(
            device), target.to(device)
        optimizer.zero_grad()
        output, out = model(data)
        out = torch.argmax(out, dim=1).float()

        CE = lossfunction(output, target)*ce
        MSE = mseloss(out, target)*mse
        L1 = l1loss(out, target)*l1
        predict_score = torch.tensor(
            [labels[x] for x in torch.argmax(output, dim=1)])
        rmse += compute_rmse_torch(score, predict_score)
        test_CE_loss += CE.item()
        test_MSE_loss += MSE.item()
        test_l1_loss += L1.item()
        loss = L1+CE+MSE
        loss.backward()
        optimizer.step()
    test_CE_loss /= len(train_loader)
    test_MSE_loss /= len(train_loader)
    test_l1_loss /= len(train_loader)
    rmse /= len(train_loader)
    if epoch % 20 == 0:
        log_message(
            'Train Epoch:{}\tCELoss: {:.4f}\tMSELoss: {:.4f}\tL1loss: {:.2f}\tAverage RMSE: {:.4f}'.format(epoch, test_CE_loss, test_MSE_loss, test_l1_loss, rmse), args)
    # if epoch % 100 == 0:
    #     print('output', out)
    #     print('target', target)
    if args.dry_run:
        return


def compute_rmse_torch(y_true, y_pred):
    # print('gt', y_true)
    # print('prediction', y_pred)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def test(model, device, test_loader, lossfunction, args):
    if args.year == 2016:
        labels = labels_2016
    elif args.year == 2021:
        labels = labels_2021
    model.eval()
    test_loss = 0
    rmse = 0
    with torch.no_grad():
        for data, target, score in test_loader:
            data, target = data.to(
                device), target.to(device)
            output = model(data)
            output, out = model(data)
            predict_score = torch.tensor(
                [labels[x] for x in torch.argmax(output, dim=1)])
            # sum up batch loss
            rmse += compute_rmse_torch(score, predict_score)
            test_loss += lossfunction(output, target).item()
    out = torch.argmax(out, dim=1).float()
    print('ValidSet output', out)
    print('ValidSet target', target)
    test_loss /= len(test_loader)
    rmse /= len(test_loader)
    log_message('Test set: Average RMSE: {:.4f}\n'.format(rmse), args)
    return rmse < best_rmse


def preprocess(train_features, train_labels,  predictyear):
    dic = {}
    testing_df = pd.read_csv(train_features)
    label = pd.read_csv(train_labels)
    testing_df = testing_df.set_index('uid')
    str_feature = []
    dic['drop_feature'] = []
    for x in testing_df.columns:
        ratio = testing_df[x].isna().sum() / len(testing_df[x])
        if ratio > 0.8:
            dic['drop_feature'].append(x)
        else:
            if testing_df[x].dtype == 'object':
                str_feature.append(x)
    df = testing_df.drop(dic['drop_feature'], axis=1)

    def colToNum(value):
        try:
            return int(str(value).split('.')[0])
        except ValueError:
            return
    for col in str_feature:
        df[col] = df[col].apply(colToNum)
    for x in df.columns:
        dic[x] = {'max': str(df[x].max()), 'min': str(df[x].min())}
        df[x] = (df[x]-df[x].min())/(df[x].max()-df[x].min())

    with open('dic.json', 'w') as f:
        json.dump(dic, f, indent=4)
    label = label[label['year'] == predictyear]
    df = df.fillna(-1)
    # df = df.dropna()
    df = df.merge(label, left_index=True, right_on='uid')
    df.to_csv(
        f"/STORAGE/peter/PREPARE/dlmodel/processed_{str(predictyear)}.csv")
    data = df.drop(
        ['year', 'uid'], axis=1)
    return data


def preprocess_onehot(train_features, train_labels,  predictyear):
    dic = {}
    testing_df = pd.read_csv(train_features)
    label = pd.read_csv(train_labels)
    testing_df = testing_df.set_index('uid')
    str_feature = []
    dic['drop_feature'] = []
    for x in testing_df.columns:
        ratio = testing_df[x].isna().sum() / len(testing_df[x])
        if ratio > 0.9:
            dic['drop_feature'].append(x)
        else:
            if testing_df[x].dtype == 'object':
                str_feature.append(x)
    testing_df = testing_df.drop(dic['drop_feature'], axis=1)
    df = pd.get_dummies(testing_df, columns=str_feature)
    for x in df.columns:
        if df[x].dtype == 'bool':
            df[x] = df[x].astype('int')
        dic[x] = {'max': str(df[x].max()), 'min': str(df[x].min())}
        df[x] = (df[x]-df[x].min())/(df[x].max()-df[x].min())
    with open('dic_onehot.json', 'w') as f:
        json.dump(dic, f, indent=4)
    label = label[label['year'] == predictyear]
    df = df.fillna(-1)
    # df = df.dropna()
    df = df.merge(label, left_index=True, right_on='uid')
    df.to_csv(
        f"/STORAGE/peter/PREPARE/dlmodel/processed_onehot_{str(predictyear)}.csv")
    data = df.drop(
        ['year', 'uid'], axis=1)
    return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PREPARE Social Determinants')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=400000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--continue-train', action='store_true',
                        help='Train from the latest status')
    parser.add_argument('--year', type=int, default=2016, metavar='N',
                        help='the survery year')
    parser.add_argument('--fold', type=int, default=0, metavar='N',
                        help='the fold to be valid set')
    parser.add_argument('--epoch', type=int, default=-1, metavar='N',)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    checkpoint_path = f'/STORAGE/peter/PREPARE/dlmodel/checkpoints_{args.year}_{args.fold}'
    startepoch = 1
    torch.manual_seed(args.seed)
    os.makedirs(checkpoint_path, exist_ok=True)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Load your dataset
    if not os.path.isfile(f'/STORAGE/peter/PREPARE/dlmodel/train_{args.year}_{args.fold}.csv') or not os.path.isfile(f'/STORAGE/peter/PREPARE/dlmodel/valid_{args.year}_{args.fold}.csv'):
        df = preprocess_onehot('/STORAGE/peter/PREPARE/train_features.csv',
                               "/STORAGE/peter/PREPARE/train_labels.csv", args.year)
        train_shuffle = shuffle(df, random_state=42)
        # Number of folds
        k = 5
        # Split the data into 5 parts (indices)
        folds = np.array_split(np.arange(len(train_shuffle)), k)
        valid_df = train_shuffle.iloc[folds[args.fold]]
        train_df = train_shuffle.iloc[np.setdiff1d(
            np.arange(len(train_shuffle)), folds[args.fold])]
        # Split the data: 80% for training, 20% for validation
        # train_df, valid_df = train_test_split(
        #     df, test_size=0.1, random_state=1, shuffle=True)
        # Save to CSV files
        train_df.to_csv(
            f'/STORAGE/peter/PREPARE/dlmodel/train_{args.year}_{args.fold}.csv', index=False)
        valid_df.to_csv(
            f'/STORAGE/peter/PREPARE/dlmodel/valid_{args.year}_{args.fold}.csv', index=False)
    train_loader = create_traindataloader(
        f"/STORAGE/peter/PREPARE/dlmodel/train_{args.year}_{args.fold}.csv", year=args.year,  batch_size=args.batch_size)
    valid_loader = create_traindataloader(
        f"/STORAGE/peter/PREPARE/dlmodel/valid_{args.year}_{args.fold}.csv", year=args.year, batch_size=args.test_batch_size, )
    # model = TransformerBinaryClassifier(in_model, 384).to(device)
    if args.year == 2016:
        out_model = len(labels_2016)
    elif args.year == 2021:
        out_model = len(labels_2021)
# one-hot process  in_model=310  otherwise  in_model=174
    model = Net(in_model=310, out_model=out_model).to(device)
    loss = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    l1loss = nn.L1Loss()
    lr = hyperparameters[f"lr_{args.year}_{args.fold}"]
    weight_decay = hyperparameters[f"weight_decay_{args.year}_{args.fold}"]
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # Load the checkpoint

    def find_largest_epoch(checkpoint_dir):
        # Get a list of all checkpoint files in the directory
        checkpoint_files = [f for f in os.listdir(
            checkpoint_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
        if len(checkpoint_path) == 0:
            return
        # Regular expression to extract the epoch number from the filename
        epoch_nums = [int(re.search(r'ckpt_(\d+).pt', f).group(1))
                      for f in checkpoint_files]

        # Find the maximum epoch number
        max_epoch = max(epoch_nums) if epoch_nums else None
        return max_epoch
    if args.continue_train:
        if args.epoch > 0:
            checkpoint = torch.load(os.path.join(
                checkpoint_path, f'ckpt_{str(args.epoch)}.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Load scheduler stat
            # scheduler.load_state_dict(
            #     checkpoint['scheduler_state_dict'])
            # Load other information like epoch and loss if available
            startepoch = checkpoint.get('epoch', 1)
            log_message(
                f"Loaded checkpoint from epoch {startepoch}.", args)
        else:
            largest_epoch = find_largest_epoch(checkpoint_path)
            if largest_epoch:
                checkpoint = torch.load(os.path.join(
                    checkpoint_path, f'ckpt_{largest_epoch}.pt'))
                # Load states for model, optimizer, and scheduler
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Load scheduler stat
                # scheduler.load_state_dict(
                #     checkpoint['scheduler_state_dict'])
                # Load other information like epoch and loss if available
                startepoch = checkpoint.get('epoch', 1)
                log_message(
                    f"Loaded checkpoint from epoch {startepoch}.", args)

    for epoch in range(startepoch, args.epochs + 1):
        train(args, model, device, train_loader,
              loss, mseloss, l1loss, optimizer, epoch)
        if epoch % 100 == 0:
            result = test(model, device, valid_loader, loss, args)
            if result:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                }, f"/STORAGE/peter/PREPARE/dlmodel/checkpoints_{args.year}_{args.fold}/best_model.pt")

            if args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                }, f"/STORAGE/peter/PREPARE/dlmodel/checkpoints_{args.year}_{args.fold}/ckpt_{epoch}.pt")
        # scheduler.step()


if __name__ == '__main__':
    main()
