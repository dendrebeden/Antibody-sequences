#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:25 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class LSTM_Bi(nn.Module):
    def __init__(self, in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len):
        super(LSTM_Bi, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(in_dim, embedding_dim)
        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.max_pool = nn.AdaptiveMaxPool1d(hidden_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fixed_len = fixed_len
        self.forward = self.forward_flen if fixed_len else self.forward_vlen
        
    def forward_flen(self, Xs, _aa2id):
        batch_size = len(Xs)

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + seq[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + seq[::-1][:-1] for seq in Xs]
        
        # list to *.tensor
        Xs_f = torch.tensor(Xs_f, device=self.device)
        Xs_b = torch.tensor(Xs_b, device=self.device)
        
        # embedding
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)
        
        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        # lstm
        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        lstm_out_valid = lstm_out_f + lstm_out_b       
        
        # pooling results from lstm
        avg_pool = torch.mean(lstm_out_valid, 1)
        max_pool, _ = torch.max(lstm_out_valid, 1)
        concat = torch.cat((avg_pool, max_pool), 1)    

        # lstm hidden state to output space
        out = F.relu(self.fc1(concat))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.reshape(batch_size, -1)

        # compute scores
        scores = F.log_softmax(out, dim=1)
        
        return scores  

    def forward_vlen(self, Xs, _aa2id):
        batch_size = len(Xs)

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + seq[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + seq[::-1][:-1] for seq in Xs]
        
        # get sequence lengths
        Xs_len = [len(seq) for seq in Xs_f]
        lmax = max(Xs_len)
        
        # list to *.tensor
        Xs_f = [torch.tensor(seq, device='cpu') for seq in Xs_f]
        Xs_b = [torch.tensor(seq, device='cpu') for seq in Xs_b]
        
        # padding
        Xs_f = pad_sequence(Xs_f, batch_first=True).to(self.device)
        Xs_b = pad_sequence(Xs_b, batch_first=True).to(self.device)
        
        # embedding
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)
        
        # packing the padded sequences
        Xs_f = pack_padded_sequence(Xs_f, Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, Xs_len, batch_first=True, enforce_sorted=False)
        
        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)
        
        # unpack outputs
        lstm_out_f, _ = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _            = pad_packed_sequence(lstm_out_b, batch_first=True)
        
        lstm_out_valid = lstm_out_f + lstm_out_b
        
        # pooling results from lstm
        avg_pool = torch.mean(lstm_out_valid, 1)
        max_pool, _ = torch.max(lstm_out_valid, 1)
        concat = torch.cat((avg_pool, max_pool), 1)    

        # lstm hidden state to output space
        out = F.relu(self.fc1(concat))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.reshape(batch_size, -1)

        # compute scores
        scores = F.log_softmax(out, dim=1)
        
        return scores  
    
    def set_param(self, param_dict):
        try:
            for pn, _ in self.named_parameters():
                exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            self.hidden_dim = param_dict['hidden_dim']
            self.fixed_len = param_dict['fixed_len']
            self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')      
    
    def get_param(self):
        param_dict = {}
        for pn, pv in self.named_parameters():
            param_dict[pn] = pv.data.cpu().numpy()
        param_dict['hidden_dim'] = self.hidden_dim
        param_dict['fixed_len'] = self.fixed_len
        return param_dict