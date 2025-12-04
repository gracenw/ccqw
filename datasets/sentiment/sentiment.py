import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from os.path import isdir
from os import listdir


## ignore warnings
import warnings
warnings.filterwarnings("ignore")


class SentimentDataset(Dataset):
    def __init__(self, split):
        self.tweets = []
        self.labels = []
        curr_path = os.path.join(os.getcwd(), 'data', 'sentiment')
        for file_name in listdir(curr_path):
            if file_name.endswith(f'{split}.txt'):
                curr_file = os.path.join(curr_path, file_name)
                with open(curr_file) as f:
                    for line in f.readlines():
                        label = int(line.split()[0])
                        tweet = ' '.join(line.split()[1:])
                        self.tweets.append(tweet)
                        self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'tweets': self.tweets[idx], 'labels': self.labels[idx]}
        # sample = {'tweets': self.tokenizer(preprocess(self.tweets[idx]), padding='longest')['input_ids'], 'labels': self.labels[idx]}
        return sample
