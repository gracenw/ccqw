import os, torch


class OneStopEnglishDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_train_data(root):
    train_texts_loc = root + '/processed/train/texts.txt'
    train_labels_loc = root + '/processed/train/labels.txt' 
    texts = []
    labels = []
    with open(train_texts_loc, 'r') as file:
        texts = [line.rstrip() for line in file]
    with open(train_labels_loc, 'r') as file:
        labels = [int(line.rstrip() )for line in file]
    return texts, labels

def read_test_data(root):
    test_texts_loc = root + '/processed/test/texts.txt'
    test_labels_loc = root + '/processed/test/labels.txt' 
    texts = []
    labels = []
    with open(test_texts_loc, 'r') as file:
        texts = [line.rstrip() for line in file]
    with open(test_labels_loc, 'r') as file:
        labels = [int(line.rstrip()) for line in file]
    return texts, labels
