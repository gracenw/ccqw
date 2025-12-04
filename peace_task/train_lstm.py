import torch
import torch.nn as nn
from torch.nn import BCELoss
from tqdm import tqdm
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from data import read_train_data, OneStopEnglishDataset
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
import string
import math
import pickle


def batch_sample(sample, size):
    segment = sample['input_ids'].squeeze()
    tag = sample['labels'].squeeze()
    start = 0
    batches = []
    while start < len(segment):
        if start + size < len(segment):
            batches.append((segment[start:start+size], tag))
        else:
            batches.append((segment[start:], tag))
        start += size
    return batches


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def main():
    TRAIN = False

    texts, labels = read_train_data('/home/gracen/repos/peace/data')

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    def prepare_sequence(seq, to_ix):
        seq = seq.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')
        idxs = []
        for w in seq:
            if w in to_ix:
                idxs.append(to_ix[w])
        return torch.tensor(idxs, dtype=torch.long)

    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64

    word_to_ix = {}
    if not TRAIN:   
        with open('/home/gracen/repos/peace/models/lstm/word_to_ix.pickle', 'rb') as f:
            word_to_ix = pickle.load(f)
        model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 2)
        model.load_state_dict(torch.load('/home/gracen/repos/peace/models/lstm/trained_mod.pth')['model_state_dict'])
    else:
        for snippet in train_texts:
            for word in snippet.translate(str.maketrans('', '', string.punctuation)).strip().split(' '):
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        with open('/home/gracen/repos/peace/models/lstm/word_to_ix.pickle', 'wb') as f:
            pickle.dump(word_to_ix, f, pickle.HIGHEST_PROTOCOL)
        model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 2)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    criterion = nn.NLLLoss()

    if TRAIN:
        for i in range(len(train_texts)):
            train_texts[i] = prepare_sequence(train_texts[i], word_to_ix)
        train_texts = {'input_ids': train_texts}
        train_dataset = OneStopEnglishDataset(train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        model.train()
        num_epochs = 100
        progress_bar = tqdm(range(len(train_loader) * num_epochs))
        optimizer = SGD(model.parameters(), lr=0.1)
        
        for epoch in range(num_epochs):
            for sample in train_loader:
                progress_bar.update(1)
                batches = batch_sample(sample, EMBEDDING_DIM)
                for segment, tag in batches:
                    optimizer.zero_grad()
                    sentence = segment.type(torch.LongTensor).to(device)
                    tag = tag.type(torch.LongTensor).to(device)
                    tag_scores = torch.squeeze(model(sentence))
                    loss = criterion(tag_scores, torch.Tensor([tag.item()] * tag_scores.shape[0], device=device).type(torch.LongTensor))
                    loss.backward()
                    optimizer.step()

        print('loss:', loss.item())
        torch.save(
            {
                'model_state_dict':  model.state_dict(),
                'loss':              loss,
            },
            f'/home/gracen/repos/peace/models/lstm/trained_mod.pth'
        )

    for i in range(len(val_texts)):
        val_texts[i] = prepare_sequence(val_texts[i], word_to_ix)
    val_texts = {'input_ids': val_texts}
    val_dataset = OneStopEnglishDataset(val_texts, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    
    model.eval()
    progress_bar = tqdm(range(len(val_loader)))
    num_correct = 0
    num_predictions = 0

    with torch.no_grad():
        for sample in val_loader:
            progress_bar.update(1)
            batches = batch_sample(sample, EMBEDDING_DIM)
            for segment, tag in batches:
                sentence = segment.type(torch.LongTensor).to(device)
                tag = tag.type(torch.LongTensor).to(device)
                tag_scores = torch.squeeze(model(sentence))
                labels = torch.Tensor([tag.item()] * tag_scores.shape[0], device=device).type(torch.LongTensor)
                loss = criterion(tag_scores, labels)
                predictions = tag_scores.argmax(-1)
                num_correct += torch.sum(predictions == labels)
                try:
                    num_predictions += len(predictions)
                except TypeError:
                    num_predictions += 1

    print('loss:', loss.item())
    print('accuracy:', (num_correct / num_predictions).item())


if __name__ == '__main__':
    main()