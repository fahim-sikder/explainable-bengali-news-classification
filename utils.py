import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifyNews(nn.Module):

    def __init__(self, bert):
        
        super(ClassifyNews, self).__init__()

        self.bert = bert

        self.dropout = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(1024, 128)

        self.fc2 = nn.Linear(128, 9)  

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        x = self.fc1(out[1])
        
        x = self.relu(x)
        
        x = self.fc2(self.dropout(x))
        
        return x


class NewsDatasets(torch.utils.data.Dataset):
    
    def __init__(self, data, max_length=100):
        
        self.data = data

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        value = self.data.iloc[idx]
        
        final_x = f'{value.title}[SEP]{value.feature}'
        
        return final_x , value['label']