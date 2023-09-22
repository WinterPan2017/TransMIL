import pandas as pd
import os
import pickle


df = pd.read_csv("/home/pwt/MIL/PosMIL/datasets/camelyon16_resnet50_10fold.csv")
train = []
val = []
test = []

for i, row in df.iterrows():
    if row[2] in [0]:
        test.append([row[0], row[1]])
    elif row[2] in [1]:
        val.append([row[0], row[1]])
    else:
        train.append([row[0], row[1]])

train_df = pd.DataFrame(data=train, columns = ['train','train_label'])
val_df = pd.DataFrame(data=val, columns = ['val','val_label'])
test_df = pd.DataFrame(data=test, columns = ['test','test_label'])

all = pd.concat([train_df, val_df, test_df], axis=1)
all.to_csv("/home/pwt/MIL/TransMIL/dataset_csv/camelyon16/camelyon16_resnet50_10fold_1.csv")