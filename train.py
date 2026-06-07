import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from transformers import BertModel, BertTokenizer

import time

from tqdm import tqdm

from utils import *

    
def train(train_data):
    
    train_data = NewsDatasets(train_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    ## Bert weight

    bert_model_name = "csebuetnlp/banglabert_large"
    
    bert = BertModel.from_pretrained(bert_model_name)
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ClassifyNews(bert)

    model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


    epochs = 10

    token_config = {

        "max_length": 100,
        "padding": "max_length",
        "return_tensors": "pt",
        "truncation": True,
        "add_special_tokens": True

    }


    for running_epoch in tqdm(range(epochs)):

        train_loss = 0

        for i, data in enumerate(tqdm(train_loader)):

            text, label = data

            model.zero_grad()

            batched_text = tokenizer.batch_encode_plus(text, **token_config)

            batched_text = batched_text.to(device)

            label = label.to(device)

            output = model(**batched_text)

            loss = criterion(output, label)

            train_loss += loss.item() * label.size(0)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if(i%len(train_loader) == 0):

                print(f'Epoch {running_epoch+1}: Training loss: {train_loss/len(train_loader)}')

        scheduler.step()

    torch.save(model.state_dict(), f'saved_weights/{time.time()}-news_large.pth')
    
    print(f'Model Saved!')

if __name__ == "__main__":
    
    data = pd.read_csv(f'data/cleaned_data.csv')
    
    train_data, _ = train_test_split(data, test_size = 0.2, random_state = 2023, stratify = data['label'])
    
    train(train_data)
