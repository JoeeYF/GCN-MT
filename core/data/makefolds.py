import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
# df = pd.read_csv('/home/yuanfang/Dataset/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')

# df['label'] = 0
# df['fold'] = 0

# for index, row in df.iterrows():
#     label_vector = np.array(row[1:])
#     label = np.argmax(label_vector)
#     df.loc[index, 'label'] = label

# kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=2020)

# for ifold,(train_idx,val_idx) in enumerate(kfold.split(df.image,df.label)):
#     df.loc[val_idx,'fold']=ifold

# # print(df.shape)
# train_df = df[df['fold'].isin(list(range(7)))]
# val_df = df[df['fold'].isin([7])]
# test_df = df[df['fold'].isin([8,9])]
# train_df.to_csv('train_index.csv',index=False)
# val_df.to_csv('val_index.csv',index=False)
# test_df.to_csv('test_index.csv',index=False)

train_df = pd.read_csv('core/data/train_index.csv')
val_df = pd.read_csv('core/data/val_index.csv')
test_df = pd.read_csv('core/data/test_index.csv')
train_df['fold_label']=0
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2020)

for ifold,(train_idx,val_idx) in enumerate(kfold.split(train_df.image,train_df.label)):
    train_df.loc[val_idx,'fold_label']=int(ifold)

train_df.to_csv('core/data/train_index.csv')
# print(train_df.shape,val_df.shape,test_df.shape)
# print(train_df.label.value_counts().sum()+val_df.label.value_counts().sum()+test_df.label.value_counts().sum())
print(train_df.label.value_counts())
# print(val_df.label.value_counts())
# print(test_df.label.value_counts())