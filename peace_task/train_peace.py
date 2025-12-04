from pathlib import Path
import math
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from os.path import isdir
from os import mkdir
from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from datasets.humaid import HumAIDDataset
from datasets.sentiment import SentimentDataset
# from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from string import ascii_lowercase
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer


## download nltk stuff for lemmatization
nltk.download('averaged_perceptron_tagger') #, download_dir='/dccstor/epochs/gwallace/nltk_cache')
nltk.download('wordnet') #, download_dir='/dccstor/epochs/gwallace/nltk_cache')
nltk.download('punkt') #, download_dir='/dccstor/epochs/gwallace/nltk_cache')
nltk.download('stopwords') #, download_dir='/dccstor/epochs/gwallace/nltk_cache')


sys.path.append(os.getcwd())
os.environ['HF_HOME'] = '/dccstor/epochs/gwallace/hf_cache'


def preprocess(tweet):
    ## remove retweet formatting
    if tweet[0:2] == 'RT':
        if ':' in tweet:
            colon = tweet.index(':')
            tweet = tweet[colon+1:]
        else:
            tweet = tweet[2:]
    ## remove whitespace
    tweet = tweet.strip()
    ## remove newlines
    tweet = tweet.replace('\n', '')
    ## set all letters to lowercase
    tweet = tweet.lower()
    ## remove url data
    tweet = re.compile(r'https?://\S+').sub('', tweet)
    ## remove ampersands
    tweet = tweet.replace('&amp;', '')
    ## remove usernamers
    tweet = ' '.join([word for word in tweet.split() if word[0] != '@'])
    ## remove all but letters and spaces from tweet
    tweet = ''.join(c for c in tweet if c in set(ascii_lowercase + ' '))
    ## tokenize words for easier processing
    tweet = word_tokenize(tweet)
    ## remove stop words
    tweet = [word for word in tweet if word not in set(stopwords.words('english'))]
    ## stemming
    stemmer = PorterStemmer()
    tweet = [stemmer.stem(word) for word in tweet]
    ## lemmatization
    def convert_to_wordnet(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    lemmatizer = WordNetLemmatizer()
    tweet = [lemmatizer.lemmatize(token, convert_to_wordnet(token)) for token in tweet]
    return ' '.join(tweet)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    learning_rate = 1e-3
    batch_size = 16
    weight_decay = 1e-4
    start_epoch = 0
    num_epochs = 1
    num_classes = 2
    checkpoint = ''
    output_dir = f'{os.getcwd()}/weights/sentiment/bert'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # tokenizer = AlbertTokenizer.from_pretrained(
    #     pretrained_model_name_or_path="albert-base-v2"
    # )
    # model = AlbertForSequenceClassification.from_pretrained(
    #     pretrained_model_name_or_path="albert-base-v2", 
        # problem_type="multi_label_classification", 
    #     num_labels=num_classes
    # ).to(device)
    tokenizer = BertTokenizer.from_pretrained(
        "google-bert/bert-base-uncased"
    )
    model = BertForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        num_labels=num_classes,
    ).to(device)
    # criterion = BCEWithLogitsLoss()
    criterion = CrossEntropyLoss()

    optimizer = AdamW(
        params=model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    lr_scheduler = StepLR(
        optimizer=optimizer, 
        step_size=10
    )

    # train_dataset = HumAIDDataset(
    #     split='train',
    # )
    train_dataset = SentimentDataset(
        split='train',
    )
    # train_dataset = load_dataset(
    #     "community-datasets/disaster_response_messages", 
    #     split='train'
    # )
    # train_batch_sampler = BatchSampler(
    #     RandomSampler(train_dataset), 
    #     batch_size, 
    #     drop_last=True
    # )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        # batch_sampler=train_batch_sampler,
        num_workers=2
    )

    output_dir = Path(output_dir)

    if checkpoint:
        output_dir = Path('/' + '/'.join(checkpoint.split('/')[:-1]))
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
    else:
        if not isdir(output_dir):
            # mkdir(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True) 
        last_version = -1
        for path in os.listdir(output_dir):
            if 'version_' in path:
                version_num = (int) (path.split('_')[-1])
                last_version = version_num if version_num > last_version else last_version
        output_dir = Path(f'{output_dir}/version_{last_version + 1:03}')
        # mkdir(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True) 

    model.train()
    criterion.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        iteration = 0
        # for batch_num, batch in tqdm(train_dataloader, total=len(train_dataloader)):
        for i, batch in enumerate(tqdm(train_dataloader)):
            tweets, labels = batch['tweets'], batch['labels']
            tweets = [preprocess(tweet) for tweet in tweets]
            # labels = F.one_hot(torch.tensor(labels, device=device), num_classes=num_classes).float()
            labels = torch.tensor(labels, device=device)
            inputs = tokenizer(tweets, return_tensors="pt", padding='longest').to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss_value = loss.item()
            total_loss += loss_value
            iteration += 1

            if not math.isfinite(loss_value):
                print(f'loss is infinite, stopping training')
                sys.exit(1)
            if loss_value > total_loss / iteration and iteration > 100:
                print(f'loss is increasing, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(f'{output_dir}/training_log.txt', 'a+') as f:
                    f.write(f'training loss: {total_loss / iteration}\n')
            

        lr_scheduler.step(epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, Path(f'{output_dir}/checkpoint.pth'))


if __name__ == '__main__':
    main()