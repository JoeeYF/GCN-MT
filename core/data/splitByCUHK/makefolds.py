import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

train_df = pd.read_csv('core/data/splitByCUHK/training.csv')
val_df = pd.read_csv('core/data/splitByCUHK/validation.csv')
test_df = pd.read_csv('core/data/splitByCUHK/testing.csv')

for index, row in train_df.iterrows():
    label_vector = np.array(row[1:-1])
    label = np.argmax(label_vector)
    train_df.loc[index, 'label'] = int(label)

for index, row in val_df.iterrows():
    label_vector = np.array(row[1:-1])
    label = np.argmax(label_vector)
    val_df.loc[index, 'label'] = int(label)

for index, row in test_df.iterrows():
    label_vector = np.array(row[1:-1])
    label = np.argmax(label_vector)
    test_df.loc[index, 'label'] = int(label)

train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=2020)

for ifold,(train_idx,val_idx) in enumerate(kfold.split(train_df.image,train_df.label)):
    train_df.loc[val_idx,'fold']=int(ifold)

train_df['fold'] = train_df['fold'].astype(int)


train_df.to_csv('core/data/splitByCUHK/train_index.csv',index=False)
val_df.to_csv('core/data/splitByCUHK/val_index.csv',index=False)
test_df.to_csv('core/data/splitByCUHK/test_index.csv',index=False)

# print(train_df.shape,val_df.shape,test_df.shape)
print(train_df[train_df['fold']==0].label.value_counts())
print(train_df[train_df['fold']==4].label.value_counts())
# print(val_df.label.value_counts())
# print(test_df.label.value_counts())