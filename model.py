#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:06 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import os
import torch
import sys

import numpy as np

from tqdm import tqdm

from lstm_bi import LSTM_Bi
from cnn import CNN
from utils_data import ProteinSeqClsDataset, aa2id_i, collate_fn_cls


class ModelBase:
    def __init__(self, model, aa2id_i, embedding_dim=64, hidden_dim=64, out_dim=2, device='cpu', gapped=True, fixed_len=True):
        self.gapped = gapped
        self.aa2id_i = aa2id_i
        in_dim = len(aa2id_i[gapped])
        self.nn = model(in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len)
        self.to(device)

    def fit(self, trn_fn, vld_fn, n_epoch=10, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None):
        # loss function and optimization algorithm
        loss_fn = torch.nn.CrossEntropyLoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # to track minimum validation loss
        min_loss = np.inf
        
        # dataset and dataset loader
        trn_data = ProteinSeqClsDataset(trn_fn, self.gapped)
        vld_data = ProteinSeqClsDataset(vld_fn, self.gapped)
        if trn_batch_size == -1: trn_batch_size = len(trn_data)
        if vld_batch_size == -1: vld_batch_size = len(vld_data)
        trn_dataloader = torch.utils.data.DataLoader(trn_data, trn_batch_size, True, collate_fn=collate_fn_cls)
        vld_dataloader = torch.utils.data.DataLoader(vld_data, vld_batch_size, False, collate_fn=collate_fn_cls)
        
        for epoch in range(n_epoch):
            # training
            self.nn.train()
            loss_avg, acc_avg, cnt = 0, 0, 0
            with tqdm(total=len(trn_data), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for features, target in trn_dataloader:
                    target = torch.tensor(target, device=self.nn.device)
                    
                    # forward and backward routine
                    self.nn.zero_grad()
                    scores = self.nn(features, self.aa2id_i[self.gapped])
                    loss = loss_fn(scores, target)
                    loss.backward()
                    op.step()
                    
                    # compute statistics
                    L = len(target)
                    predicted = torch.argmax(scores, 1)
                    loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                    corr = (predicted == target).data.cpu().numpy()
                    acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                    cnt += L
                    
                    # update progress bar
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
                    pbar.update(len(features))
            
            # validation
            self.nn.eval()
            loss_avg, acc_avg, cnt = 0, 0, 0
            with torch.set_grad_enabled(False):
                with tqdm(total=len(vld_data), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                    for features, target in vld_dataloader:
                        target = torch.tensor(target, device=self.nn.device)
                        
                        # forward routine
                        scores = self.nn(features, self.aa2id_i[self.gapped])
                        loss = loss_fn(scores, target)
                        
                        # compute statistics
                        L = len(target)
                        predicted = torch.argmax(scores, 1)
                        loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                        corr = (predicted == target).data.cpu().numpy()
                        acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                        cnt += L
                        
                        # update progress bar
                        pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
                        pbar.update(len(target))
            
            # save model
            if loss_avg < min_loss and save_fp:
                min_loss = loss_avg
                model_name = type(self.nn).__name__
                os.makedirs(save_fp, exist_ok=True)
                self.save('{}/{}_{:.6f}.npy'.format(save_fp, model_name, loss_avg))
    
    def eval(self, fn, batch_size=512):        
        # dataset and dataset loader
        data = ProteinSeqClsDataset(fn, self.gapped)
        if batch_size == -1: batch_size = len(data)
        dataloader = torch.utils.data.DataLoader(data, batch_size, False, collate_fn=collate_fn_cls)
        
        self.nn.eval()
        scores = np.zeros(len(data), dtype=np.float32)
        sys.stdout.flush()
        with torch.set_grad_enabled(False):
            with tqdm(total=len(data), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for n, (features, target) in enumerate(dataloader):
                    out = self.nn(features, self.aa2id_i[self.gapped])
                    predicted = torch.argmax(out, 1)
                    scores[n*batch_size:(n+1)*batch_size] = predicted

                    pbar.update(len(target))
        return scores
    
    def save(self, fn):
        param_dict = self.nn.get_param()
        param_dict['gapped'] = self.gapped
        np.save(fn, param_dict)
    
    def load(self, fn):
        param_dict = np.load(fn, allow_pickle=True).item()
        self.gapped = param_dict['gapped']
        self.nn.set_param(param_dict)

    def to(self, device):
        self.nn.to(device)
        self.nn.device = device
        
    def summary(self):
        for n, w in self.nn.named_parameters():
            print('{}:\t{}'.format(n, w.shape))
        print('Fixed Length:\t{}'.format(self.nn.fixed_len) )
        print('Gapped:\t{}'.format(self.gapped))
        print('Device:\t{}'.format(self.nn.device))


class ModelLSTM(ModelBase):
    def __init__(self, embedding_dim=64, hidden_dim=64, out_dim=2, device='cpu', gapped=True, fixed_len=True):
        super().__init__(LSTM_Bi, aa2id_i, embedding_dim, hidden_dim, out_dim, device, gapped, fixed_len)
        
class ModelCNN(ModelBase):
    def __init__(self, embedding_dim=64, hidden_dim=64, out_dim=2, device='cpu', gapped=True, fixed_len=True):
        super().__init__(CNN, aa2id_i, embedding_dim, hidden_dim, out_dim, device, gapped, fixed_len)