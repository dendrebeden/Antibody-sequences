#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:25 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CNN(nn.Module):
# "shallow" CNN implementation
# TO DO: implement deep CNN
    def __init__(self, in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len):
        super(CNN, self).__init__()
        filter_sizes = [1,2,3,5]
        self.device = device
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(in_dim, embedding_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, hidden_dim, (K, embedding_dim)) for K in filter_sizes])
        self.fc1 = nn.Linear(hidden_dim * len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fixed_len = fixed_len
        self.forward = self.forward_flen if fixed_len else self.forward_vlen
        
    def forward_flen(self, Xs, _aa2id):
        # pad <EOS> & <SOS>
        Xs = [[_aa2id['<SOS>']] + seq[:-1] + [_aa2id['<EOS>']] for seq in Xs]
        
        # list to *.tensor
        Xs = torch.tensor(Xs, device=self.device)

        # embedding
        Xs = self.word_embeddings(Xs)
        Xs = Xs.unsqueeze(1)  
    
        Xs = [F.relu(conv(Xs)).squeeze(3) for conv in self.convs1]
        Xs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in Xs]

        Xs = torch.cat(Xs, 1)

        # lstm hidden state to output space
        out = F.relu(self.fc1(Xs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # compute scores
        scores = F.log_softmax(out, dim=1)
        
        return scores

    def forward_vlen(self, Xs, _aa2id):
        # pad <EOS> & <SOS>
        Xs = [[_aa2id['<SOS>']] + seq[:-1] + [_aa2id['<EOS>']] for seq in Xs]
        
        # list to *.tensor
        Xs = [torch.tensor(seq, device='cpu') for seq in Xs]
        
        # padding
        Xs = pad_sequence(Xs, batch_first=True).to(self.device)
        
        # embedding
        Xs = self.word_embeddings(Xs)
        Xs = Xs.unsqueeze(1)  
    
        Xs = [F.relu(conv(Xs)).squeeze(3) for conv in self.convs1]
        Xs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in Xs]  

        Xs = torch.cat(Xs, 1)

        # lstm hidden state to output space
        out = F.relu(self.fc1(Xs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # compute scores
        scores = F.log_softmax(out, dim=1)
        
        return scores
    
    def set_param(self, param_dict):
        try:
            for pn, _ in self.named_parameters():
                # a dirty workaround!
                if not pn.startswith("convs1"):
                    exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            self.convs1._modules = param_dict["convs1"]
            self.hidden_dim = param_dict['hidden_dim']
            self.fixed_len = param_dict['fixed_len']
            self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')      
    
    def get_param(self):
        param_dict = {}
        for pn, pv in self.named_parameters():
            # a dirty workaround!
            if not pn.startswith("convs1"):
                param_dict[pn] = pv.data.cpu().numpy()
        param_dict['convs1'] = self.convs1._modules
        param_dict['hidden_dim'] = self.hidden_dim
        param_dict['fixed_len'] = self.fixed_len
        return param_dict
        