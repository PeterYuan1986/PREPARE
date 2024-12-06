import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from network import Net, TransformerBinaryClassifier
from dataloader import create_traindataloader
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np


def train(args, model, device, train_loader, lossfunction, optimizer, epoch):
    model.train()
    test_loss = 0
    rmse = 0
    for batch_idx, (data, target, score) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # if batch_idx == 0:
        #     print('output', torch.argmax(output, dim=1))
        #     print('target', target)
        loss = lossfunction(output, target)
        predict_score = torch.tensor(
            [list(range(5, 385, 5))[x] for x in torch.argmax(output, dim=1)])
        rmse += compute_rmse_torch(score, predict_score)
        test_loss += loss.item()
        loss.backward()
        optimizer.step()
    test_loss /= len(train_loader)
    rmse /= len(train_loader)
    print(
        '\nTrain Epoch:{}\tLoss: {:.6f}\tAverage RMSE: {:.4f}\n'.format(epoch, test_loss,  rmse))
    if args.dry_run:
        return


def compute_rmse_torch(y_true, y_pred):
    # print('gt', y_true)
    # print('prediction', y_pred)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def test(model, device, test_loader, lossfunction):
    model.eval()
    test_loss = 0
    rmse = 0
    with torch.no_grad():
        for data, target, score in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predict_score = torch.tensor(
                [list(range(5, 385, 5))[x] for x in torch.argmax(output, dim=1)])
            # sum up batch loss
            rmse += compute_rmse_torch(score, predict_score)
            test_loss += lossfunction(output, target).item()
    test_loss /= len(test_loader)
    rmse /= len(test_loader)
    print(
        '\nTest set: Average RMSE: {:.4f}\n'.format(rmse))


# def preprocess(csv_file, label_file, predictyear):
#     df = pd.read_csv(csv_file)
#     label = pd.read_csv(label_file)
#     label = label[label['year'] == predictyear]
#     features_03 = ['age_03', 'urban_03', 'married_03', 'n_mar_03', 'edu_gru_03', 'n_living_child_03', 'migration_03', 'glob_hlth_03', 'adl_dress_03', 'adl_walk_03', 'adl_bath_03', 'adl_eat_03', 'adl_bed_03', 'adl_toilet_03', 'n_adl_03', 'iadl_money_03', 'iadl_meds_03', 'iadl_shop_03', 'iadl_meals_03', 'n_iadl_03', 'depressed_03', 'hard_03', 'restless_03', 'happy_03', 'lonely_03', 'enjoy_03', 'sad_03', 'tired_03', 'energetic_03', 'n_depr_03', 'cesd_depressed_03', 'hypertension_03', 'diabetes_03', 'resp_ill_03', 'arthritis_03', 'hrt_attack_03', 'stroke_03', 'cancer_03', 'n_illnesses_03', 'bmi_03', 'exer_3xwk_03', 'alcohol_03', 'tobacco_03', 'test_chol_03', 'test_tuber_03', 'test_diab_03', 'test_pres_03', 'hosp_03', 'visit_med_03', 'out_proc_03', 'visit_dental_03', 'imss_03', 'issste_03', 'pem_def_mar_03', 'insur_private_03', 'insur_other_03', 'insured_03', 'decis_famil_03', 'decis_personal_03', 'employment_03', 'sgender_03', 'rjob_hrswk_03',
#                    # 'rjlocc_m_03', 'rjob_end_03', 'rjobend_reason_03',
#                    'rearnings_03', 'searnings_03', 'hincome_03', 'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03', 'hinc_cap_03', 'rinc_pension_03', 'sinc_pension_03', 'rrelgimp_03']

#     features_12 = ['age_12', 'urban_12', 'married_12', 'n_mar_12', 'edu_gru_12', 'n_living_child_12',
#                    'migration_12', 'glob_hlth_12', 'adl_dress_12', 'adl_walk_12', 'adl_bath_12', 'adl_eat_12', 'adl_bed_12', 'adl_toilet_12', 'n_adl_12',
#                    'iadl_money_12', 'iadl_meds_12', 'iadl_shop_12', 'iadl_meals_12', 'n_iadl_12', 'depressed_12', 'hard_12', 'restless_12', 'happy_12', 'lonely_12', 'enjoy_12', 'sad_12', 'tired_12', 'energetic_12', 'n_depr_12', 'cesd_depressed_12', 'hypertension_12', 'diabetes_12', 'resp_ill_12', 'arthritis_12', 'hrt_attack_12', 'stroke_12', 'cancer_12', 'n_illnesses_12', 'bmi_12', 'exer_3xwk_12', 'alcohol_12', 'tobacco_12', 'test_chol_12', 'test_tuber_12', 'test_diab_12', 'test_pres_12', 'hosp_12', 'visit_med_12', 'out_proc_12', 'visit_dental_12',
#                    'imss_12', 'issste_12', 'pem_def_mar_12', 'insur_private_12', 'insur_other_12', 'seg_pop_12', 'insured_12', 'decis_famil_12', 'decis_personal_12', 'employment_12', 'vax_flu_12', 'vax_pneu_12', 'care_adult_12', 'care_child_12', 'volunteer_12', 'attends_class_12', 'attends_club_12', 'reads_12', 'games_12', 'table_games_12', 'comms_tel_comp_12', 'act_mant_12', 'tv_12', 'sewing_12', 'satis_ideal_12', 'satis_excel_12', 'satis_fine_12', 'cosas_imp_12', 'wouldnt_change_12', 'memory_12', 'sgender_12',
#                    #    # 'rjob_hrswk_12','rjlocc_m_12', 'rjob_end_12', 'rjobend_reason_12',
#                    'rearnings_12',
#                    #    #  'searnings_12',
#                    'hincome_12', 'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12', 'hinc_cap_12', 'rinc_pension_12', 'sinc_pension_12', 'rrelgimp_12', 'rrfcntx_m_12', 'rsocact_m_12', 'rrelgwk_12',
#                    #    #    'a16a_12', 'a21_12', 'a22_12', 'a33b_12',
#                    'a34_12',
#                    #    #    'j11_12'
#                    ]

#     def colToNum(value):
#         try:
#             return int(str(value).split('.')[0])
#         except ValueError:
#             return

#     for col in features_03+features_12:
#         if col not in df.columns:
#             df[col] = -1
#         else:
#             if df[col].dtypes == np.object_:
#                 df[col] = df[col].apply(colToNum)
#     df = df[['uid']+features_03+features_12]
#     df = df.dropna(subset=['age_03'])
#     df = df.fillna(-1)
#     df = df.dropna()
#     df = df.merge(label, left_on='uid', right_on='uid')
#     df.to_csv('test.csv')
#     data = df.drop(
#         ['year', 'uid'], axis=1)
#     return data

def preprocess_onehot(csv_file, label_file, predictyear):
    df = pd.read_csv(csv_file)
    label = pd.read_csv(label_file)
    label = label[label['year'] == predictyear]
    # features_03 = ['age_03', 'urban_03', 'married_03', 'n_mar_03', 'edu_gru_03', 'n_living_child_03', 'migration_03', 'glob_hlth_03', 'adl_dress_03', 'adl_walk_03', 'adl_bath_03', 'adl_eat_03', 'adl_bed_03', 'adl_toilet_03', 'n_adl_03', 'iadl_money_03', 'iadl_meds_03', 'iadl_shop_03', 'iadl_meals_03', 'n_iadl_03', 'depressed_03', 'hard_03', 'restless_03', 'happy_03', 'lonely_03', 'enjoy_03', 'sad_03', 'tired_03', 'energetic_03', 'n_depr_03', 'cesd_depressed_03', 'hypertension_03', 'diabetes_03', 'resp_ill_03', 'arthritis_03', 'hrt_attack_03', 'stroke_03', 'cancer_03', 'n_illnesses_03', 'bmi_03', 'exer_3xwk_03', 'alcohol_03', 'tobacco_03', 'test_chol_03', 'test_tuber_03', 'test_diab_03', 'test_pres_03', 'hosp_03', 'visit_med_03', 'out_proc_03', 'visit_dental_03', 'imss_03', 'issste_03', 'pem_def_mar_03', 'insur_private_03', 'insur_other_03', 'insured_03', 'decis_famil_03', 'decis_personal_03', 'employment_03', 'sgender_03', 'rjob_hrswk_03',
    #                # 'rjlocc_m_03', 'rjob_end_03', 'rjobend_reason_03',
    #                'rearnings_03', 'searnings_03', 'hincome_03', 'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03', 'hinc_cap_03', 'rinc_pension_03', 'sinc_pension_03', 'rrelgimp_03']

    # features_12 = ['age_12', 'urban_12', 'married_12', 'n_mar_12', 'edu_gru_12', 'n_living_child_12',
    #                'migration_12', 'glob_hlth_12', 'adl_dress_12', 'adl_walk_12', 'adl_bath_12', 'adl_eat_12', 'adl_bed_12', 'adl_toilet_12', 'n_adl_12',
    #                'iadl_money_12', 'iadl_meds_12', 'iadl_shop_12', 'iadl_meals_12', 'n_iadl_12', 'depressed_12', 'hard_12', 'restless_12', 'happy_12', 'lonely_12', 'enjoy_12', 'sad_12', 'tired_12', 'energetic_12', 'n_depr_12', 'cesd_depressed_12', 'hypertension_12', 'diabetes_12', 'resp_ill_12', 'arthritis_12', 'hrt_attack_12', 'stroke_12', 'cancer_12', 'n_illnesses_12', 'bmi_12', 'exer_3xwk_12', 'alcohol_12', 'tobacco_12', 'test_chol_12', 'test_tuber_12', 'test_diab_12', 'test_pres_12', 'hosp_12', 'visit_med_12', 'out_proc_12', 'visit_dental_12',
    #                'imss_12', 'issste_12', 'pem_def_mar_12', 'insur_private_12', 'insur_other_12', 'seg_pop_12', 'insured_12', 'decis_famil_12', 'decis_personal_12', 'employment_12', 'vax_flu_12', 'vax_pneu_12', 'care_adult_12', 'care_child_12', 'volunteer_12', 'attends_class_12', 'attends_club_12', 'reads_12', 'games_12', 'table_games_12', 'comms_tel_comp_12', 'act_mant_12', 'tv_12', 'sewing_12', 'satis_ideal_12', 'satis_excel_12', 'satis_fine_12', 'cosas_imp_12', 'wouldnt_change_12', 'memory_12', 'sgender_12',
    #                #    # 'rjob_hrswk_12','rjlocc_m_12', 'rjob_end_12', 'rjobend_reason_12',
    #                'rearnings_12',
    #                #    #  'searnings_12',
    #                'hincome_12', 'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12', 'hinc_cap_12', 'rinc_pension_12', 'sinc_pension_12', 'rrelgimp_12', 'rrfcntx_m_12', 'rsocact_m_12', 'rrelgwk_12',
    #                #    #    'a16a_12', 'a21_12', 'a22_12', 'a33b_12',
    #                'a34_12',
    #                #    #    'j11_12'
    #                ]

    # def colToNum(value):
    #     try:
    #         return int(str(value).split('.')[0])
    #     except ValueError:
    #         return

    # for col in features_03+features_12:
    #     if col not in df.columns:
    #         df[col] = -1
    #     else:
    #         if df[col].dtypes == np.object_:
    #             df[col] = df[col].apply(colToNum)
    # df = df[['uid']+features_03+features_12]
    # df = df.dropna(subset=['age_03'])
    df = df.fillna(-1)
    # df = df.dropna()
    df = df.merge(label, left_on='uid', right_on='uid')
    df.to_csv('test.csv')
    data = df.drop(
        ['year', 'uid'], axis=1)
    return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PREPARE Social Determinants')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=200000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1)')
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
    parser.add_argument('--year', type=int, default=2021, metavar='N',
                        help='the survery year')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    checkpoint_path = f'/STORAGE/peter/PREPARE/dlmodel/checkpoints_{args.year}'
    startepoch = 1
    torch.manual_seed(args.seed)
    os.makedirs(checkpoint_path, exist_ok=True)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Load your dataset
    if not os.path.isfile(f'/STORAGE/peter/PREPARE/dlmodel/train_{args.year}.csv') or not os.path.isfile(f'/STORAGE/peter/PREPARE/dlmodel/valid_{args.year}.csv'):
        df = preprocess_onehot('/STORAGE/peter/PREPARE/train_features_onehot.csv',
                               "/STORAGE/peter/PREPARE/train_labels.csv", args.year)
        # Split the data: 80% for training, 20% for validation
        train_df, valid_df = train_test_split(
            df, test_size=0.2, random_state=1, shuffle=True)
        # Save to CSV files
        train_df.to_csv(
            f'/STORAGE/peter/PREPARE/dlmodel/train_{args.year}.csv', index=False)
        valid_df.to_csv(
            f'/STORAGE/peter/PREPARE/dlmodel/valid_{args.year}.csv', index=False)
    in_model = len(pd.read_csv(
        f"/STORAGE/peter/PREPARE/dlmodel/train_{args.year}.csv").columns)-1

    train_loader = create_traindataloader(
        f"/STORAGE/peter/PREPARE/dlmodel/train_{args.year}.csv", batch_size=args.batch_size)
    valid_loader = create_traindataloader(
        f"/STORAGE/peter/PREPARE/dlmodel/valid_{args.year}.csv", batch_size=args.test_batch_size)
    # model = TransformerBinaryClassifier(in_model, 384).to(device)
    model = Net(in_model, 76).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
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
            print(f"Loaded checkpoint from epoch {startepoch}.")
    for epoch in range(startepoch, args.epochs + 1):
        train(args, model, device, train_loader, loss, optimizer, epoch)
        if epoch % 100 == 0:
            test(model, device, valid_loader, loss)
            if args.save_model and epoch % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                }, f"/STORAGE/peter/PREPARE/dlmodel/checkpoints_{args.year}/ckpt_{epoch}.pt")
        # scheduler.step()


if __name__ == '__main__':
    main()
