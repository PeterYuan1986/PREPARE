import pandas as pd
from dataloader import create_testdataloader
from network import Net, TransformerBinaryClassifier
import torch
import numpy as np
import json
from dataloader import *


def test(model, device, test_loader, year):
    model.eval()
    result = []
    if year == 2016:
        labels = labels_2016
    if year == 2021:
        labels = labels_2021
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            result += [labels[x]
                       for x in torch.argmax(output, dim=1)]
            # sum up batch loss
    return result


with open('./dic.json', 'r') as f:
    val = json.load(f)


def preprocess_onehot(test_features):
    testing_df = pd.read_csv(test_features)
    testing_df = testing_df.set_index('uid')
    str_feature = []
    for x in testing_df.columns:
        if testing_df[x].dtype == 'object':
            str_feature.append(x)
    df = pd.get_dummies(testing_df, columns=str_feature)
    for x in df.columns:
        if df[x].dtype == 'bool':
            df[x] = df[x].astype('int')
        ma = float(val[x]['max'])
        mi = float(val[x]['min'])
        df[x] = (df[x]-mi)/(ma-mi)
    df = df.fillna(-1)
    a = set(df.columns)
    train = pd.read_csv('train_2021.csv')
    b = set(train.columns)
    print(b-a)
    for col in list(b-a):
        if col != 'composite_score':
            df[col] = -1
    df = df[train.columns[:-1]]
    df.to_csv('prediction.csv')
    return df


checkpoint_2016_0_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2016_0/ckpt_3220_3565_3535.pt"
checkpoint_2021_0_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2021_0/ckpt_2600_3587_3753.pt"
checkpoint_2016_1_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2016_1/ckpt_3100_3671_3662.pt"
checkpoint_2021_1_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2021_1/ckpt_2700_3747_3576.pt"
checkpoint_2016_2_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2016_2/ckpt_3100_3671_3662.pt"
checkpoint_2021_2_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2021_2/ckpt_2640_3504_3540.pt"
checkpoint_2016_3_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2016_3/ckpt_3180_3667_3661.pt"
checkpoint_2021_3_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2021_3/ckpt_2500_3673_3634.pt"
checkpoint_2016_4_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2016_4/ckpt_3580_3535_3384.pt"
checkpoint_2021_4_path = "/STORAGE/peter/PREPARE/dlmodel_v5span/checkpoints_2021_4/ckpt_2800_3638_3644.pt"


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
processed_df = preprocess_onehot("/STORAGE/peter/PREPARE/test_features.csv")
dl = create_testdataloader(processed_df)
# in_model = len(pd.read_csv(
#     f"/STORAGE/peter/PREPARE/dlmodel/train_2016.csv").columns)-1
model = Net(340, 76).to(device)
checkpoint_2016_0 = torch.load(checkpoint_2016_0_path)
checkpoint_2021_0 = torch.load(checkpoint_2021_0_path)
checkpoint_2016_1 = torch.load(checkpoint_2016_1_path)
checkpoint_2021_1 = torch.load(checkpoint_2021_1_path)
checkpoint_2016_2 = torch.load(checkpoint_2016_2_path)
checkpoint_2021_2 = torch.load(checkpoint_2021_2_path)
checkpoint_2016_3 = torch.load(checkpoint_2016_3_path)
checkpoint_2021_3 = torch.load(checkpoint_2021_3_path)
checkpoint_2016_4 = torch.load(checkpoint_2016_4_path)
checkpoint_2021_4 = torch.load(checkpoint_2021_4_path)
model.load_state_dict(checkpoint_2016_0['model_state_dict'])
processed_df['cs_2016_0'] = test(model, device, dl, 2016)
model.load_state_dict(checkpoint_2021_0['model_state_dict'])
processed_df['cs_2021_0'] = test(model, device, dl, 2021)
model.load_state_dict(checkpoint_2016_1['model_state_dict'])
processed_df['cs_2016_1'] = test(model, device, dl, 2016)
model.load_state_dict(checkpoint_2021_1['model_state_dict'])
processed_df['cs_2021_1'] = test(model, device, dl, 2021)
model.load_state_dict(checkpoint_2016_2['model_state_dict'])
processed_df['cs_2016_2'] = test(model, device, dl, 2016)
model.load_state_dict(checkpoint_2021_2['model_state_dict'])
processed_df['cs_2021_2'] = test(model, device, dl, 2021)
model.load_state_dict(checkpoint_2016_3['model_state_dict'])
processed_df['cs_2016_3'] = test(model, device, dl, 2016)
model.load_state_dict(checkpoint_2021_3['model_state_dict'])
processed_df['cs_2021_3'] = test(model, device, dl, 2021)
model.load_state_dict(checkpoint_2016_4['model_state_dict'])
processed_df['cs_2016_4'] = test(model, device, dl, 2016)
model.load_state_dict(checkpoint_2021_4['model_state_dict'])
processed_df['cs_2021_4'] = test(model, device, dl, 2021)
processed_df.to_csv('result.csv')
r_2016 = pd.DataFrame()
r_2016["uid"] = processed_df.index
r_2016["year"] = 2016
r_2016["composite_score"] = ((processed_df["cs_2016_0"]+processed_df["cs_2016_1"] +
                             processed_df["cs_2016_2"]+processed_df["cs_2016_3"]+processed_df["cs_2016_4"])/5).values
r_2021 = pd.DataFrame()
r_2021["uid"] = processed_df.index
r_2021["year"] = 2021
r_2021["composite_score"] = ((processed_df["cs_2021_0"]+processed_df["cs_2021_1"] +
                             processed_df["cs_2021_2"]+processed_df["cs_2021_3"]+processed_df["cs_2021_4"])/5).values
r = pd.concat([r_2016, r_2021])
submit = pd.read_csv(
    "/STORAGE/peter/PREPARE/submission_format.csv")
submit = pd.merge(submit, r_2016[["uid", 'year', "composite_score"]], on=[
                  "uid", 'year'], how="left")
submit = pd.merge(
    submit, r_2021[["uid", 'year', "composite_score"]], on=["uid", 'year'], how="left")
submit = submit.fillna(0)
submit['composite_score'] = (
    submit['composite_score']+submit["composite_score_x"] + submit["composite_score_y"])

submit = submit[["uid", 'year', "composite_score"]]
submit['composite_score'] = np.ceil(submit['composite_score']).astype(int)
submit = submit.astype({'composite_score': 'int32'})
submit.to_csv('submission_format.csv', index=False)
