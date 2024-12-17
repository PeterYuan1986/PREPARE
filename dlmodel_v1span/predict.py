import pandas as pd
from dataloader import create_testdataloader
from network import Net, TransformerBinaryClassifier
import torch
import os
import json


def test(model, device, test_loader):
    model.eval()
    result = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            result += [list(range(5, 385, 5))[x]
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


checkpoint_2016_path = "/STORAGE/peter/PREPARE/dlmodel/checkpoints_2016/ckpt_9200.pt"
checkpoint_2021_path = "/STORAGE/peter/PREPARE/dlmodel/checkpoints_2021/ckpt_15800.pt"
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
processed_df = preprocess_onehot("/STORAGE/peter/PREPARE/test_features.csv")
dl = create_testdataloader(processed_df)
in_model = len(pd.read_csv(
    f"/STORAGE/peter/PREPARE/dlmodel/train_2016.csv").columns)-1
model = Net(in_model, 76).to(device)
checkpoint_2016 = torch.load(checkpoint_2016_path)
checkpoint_2021 = torch.load(checkpoint_2021_path)
model.load_state_dict(checkpoint_2016['model_state_dict'])
processed_df['cs_2016'] = test(model, device, dl)
model.load_state_dict(checkpoint_2021['model_state_dict'])
processed_df['cs_2021'] = test(model, device, dl)
r_2016 = pd.DataFrame()
r_2016["uid"] = processed_df.index
r_2016["year"] = 2016
r_2016["composite_score"] = processed_df["cs_2016"].values
r_2021 = pd.DataFrame()
r_2021["uid"] = processed_df.index
r_2021["year"] = 2021
r_2021["composite_score"] = processed_df["cs_2021"].values
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
submit = submit.astype({'composite_score': 'int32'})
submit.to_csv('submission_format.csv', index=False)
