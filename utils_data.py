#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:05:54 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import pandas as pd
import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset

# true if gapped else false
vocab_o = { True: ['-'] + ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
           False: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']}
aa2id_o = { True: dict(zip(vocab_o[True],  list(range(len(vocab_o[True]))))),
           False: dict(zip(vocab_o[False], list(range(len(vocab_o[False])))))}
id2aa_o = { True: dict(zip(list(range(len(vocab_o[True]))),  vocab_o[True])),
           False: dict(zip(list(range(len(vocab_o[False]))), vocab_o[False]))}

vocab_i = { True: vocab_o[True]  + ['<SOS>', '<EOS>'],
           False: vocab_o[False] + ['<SOS>', '<EOS>']}
aa2id_i = { True: dict(zip(vocab_i[True],  list(range(len(vocab_i[True]))))),
           False: dict(zip(vocab_i[False], list(range(len(vocab_i[False])))))}
id2aa_i = { True: dict(zip(list(range(len(vocab_i[True]))),  vocab_i[True])),
           False: dict(zip(list(range(len(vocab_i[False]))), vocab_i[False]))}

class ProteinSeqClsDataset(Dataset):
    def __init__(self, fn, gapped=True):
        # load data
        df = pd.read_csv(fn)
        
        self.x = [[aa2id_i[gapped][c] for c in r] for r in df["sequence"].values]
        self.y = df["target"].values        
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ProteinSeqDataset(Dataset):
    def __init__(self, fn, gapped=True):
        # load data
        with open(fn, 'r') as f:
            self.data = [l.strip('\n') for l in f]
        
        # char to id
        self.data = [[aa2id_i[gapped][c] for c in r] for r in self.data] 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def preprocess_classification_data(
    human_path: str,
    mouse_path: str,
    save_path: str
) -> None:
    # Data load
    human = pd.read_csv(human_path, header=None)
    mouse = pd.read_csv(mouse_path, header=None)

    human.columns = ["sequence"]
    mouse.columns = ["sequence"]

    # Map human and mouse target to 1 and 0 correspondingly
    human["target"] = 1
    mouse["target"] = 0
    
    # Human train/val/test split
    human_train, human_test = train_test_split(human,
    test_size=0.1, shuffle = True, random_state = 1)

    human_val, human_test = train_test_split(human_test, 
    test_size=0.5, random_state= 1) # 0.1 x 0.5 = 0.05
    
    # Human train/val/test split
    mouse_train, mouse_test = train_test_split(mouse,
    test_size=0.1, shuffle = True, random_state = 1)

    mouse_val, mouse_test = train_test_split(mouse_test, 
    test_size=0.5, random_state= 1) # 0.1 x 0.5 = 0.05
    
    # Concat and shuffle human/mouse data
    train = shuffle(
    pd.concat([human_train, mouse_train], ignore_index=True)
    )
    val = shuffle(
        pd.concat([human_val, mouse_val], ignore_index=True)
    )
    test = shuffle(
        pd.concat([human_test, mouse_test], ignore_index=True)
    )
    
    # Export final tables
    os.makedirs(save_path, exist_ok=True)
    train.to_csv(os.path.join(save_path, "train_vlen.txt"), index=False)
    val.to_csv(os.path.join(save_path, "val_vlen.txt"), index=False)
    test.to_csv(os.path.join(save_path, "test_vlen.txt"), index=False)
    
    return None

def collate_fn_cls(batch):
    return [x[0] for x in batch], [x[1] for x in batch]
    
def collate_fn(batch):
    return batch, [x for seq in batch for x in seq]