import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertModel, BertTokenizer

import time
from tqdm import tqdm

from utils import *
    
    
def test(test_data):

    test_data = NewsDatasets(test_data)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

    bert_model_name = "csebuetnlp/banglabert_large"
    
    bert = BertModel.from_pretrained(bert_model_name)
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ClassifyNews(bert)
    
    ## path of the saved_weights
    
    saved_model_path = f'saved_weights/1678583057.8259-news-classification-even-data-31.pth'

    checkpoint = torch.load(saved_model_path)
    
    model.load_state_dict(checkpoint)
    
    model.to(device)

    model.eval()

    token_config = {

        "max_length": 100,
        "padding": "max_length",
        "return_tensors": "pt",
        "truncation": True,
        "add_special_tokens": True

    }

    all_preds = []
    
    all_labels = []

    for i, data in enumerate(tqdm(test_loader)):
        
        text, labels = data
        
        batched_text = tokenizer.batch_encode_plus(text, **token_config)

        batched_text = batched_text.to(device)

        labels = labels.to(device)
        
        with torch.no_grad():
            
            output = model(**batched_text)
            
        preds = output.detach().cpu().numpy()
        
        preds = np.argmax(preds, axis = 1)
        
        all_preds.extend(preds)
        
        all_labels.extend(labels.cpu().numpy())
        
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    
    data = pd.read_csv(f'data/even_cleaned_data.csv')
    
    _, test_data = train_test_split(data, test_size = 0.2, random_state = 2023, stratify = data['label'])
    
    test(test_data)
