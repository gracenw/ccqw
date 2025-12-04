import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from os.path import isdir
from os import listdir


## ignore warnings
import warnings
warnings.filterwarnings("ignore")


# translate_labels = {
#     'caution_and_advice'                        : 0,
#     'displaced_people_and_evacuations'          : 1,
#     'dont_know_cant_judge'                      : 2,
#     'infrastructure_and_utility_damage'         : 3,
#     'injured_or_dead_people'                    : 4,
#     'missing_or_found_people'                   : 5,
#     'not_humanitarian'                          : 6,
#     'other_relevant_information'                : 7,
#     'requests_or_urgent_needs'                  : 8,
#     'rescue_volunteering_or_donation_effort'    : 9,
#     'sympathy_and_support'                      : 10,
# }

## 0: not urgent
## 1: slightly urgent
## 2: very urgent
translate_labels = {
    'caution_and_advice'                        : 1,
    'displaced_people_and_evacuations'          : 2,
    'dont_know_cant_judge'                      : 0,
    'infrastructure_and_utility_damage'         : 2,
    'injured_or_dead_people'                    : 2,
    'missing_or_found_people'                   : 2,
    'not_humanitarian'                          : 0,
    'other_relevant_information'                : 0,
    'requests_or_urgent_needs'                  : 1,
    'rescue_volunteering_or_donation_effort'    : 1,
    'sympathy_and_support'                      : 0,
}


class HumAIDDataset(Dataset):
    def __init__(self, split):
        self.tweets = []
        self.labels = []
        for path_name in os.listdir(f'{os.getcwd()}/data/humaid/events'):
            curr_path = os.path.join(f'{os.getcwd()}/data/humaid/events', path_name)
            if isdir(curr_path):
                for file_name in listdir(curr_path):
                    if file_name.endswith(f'_{split}.tsv'):
                        curr_file = os.path.join(f'{os.getcwd()}/data/humaid/events', path_name, file_name)
                        data = pd.read_csv(curr_file, sep='\t')
                        for tweet in data['tweet_text'].values:
                            ## clean up twitter specific formatting
                            # self.tweets.append(self.tokenizer(preprocess(tweet), padding='max_length'))
                            self.tweets.append(tweet)
                        for label in data['class_label'].values:
                            ## convert labels to number format
                            self.labels.append(translate_labels[label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'tweets': self.tweets[idx], 'labels': self.labels[idx]}
        # sample = {'tweets': self.tokenizer(preprocess(self.tweets[idx]), padding='longest')['input_ids'], 'labels': self.labels[idx]}
        return sample
