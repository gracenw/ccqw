import torch
import torch.nn as nn
from torch.nn import BCELoss
from tqdm import tqdm
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from data import read_train_data, OneStopEnglishDataset
from nltk.tokenize import word_tokenize


class Shallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


def main():
    texts, labels = read_train_data('/home/gracen/repos/peace/data')

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)

    train_dataset = OneStopEnglishDataset(train_encodings, train_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Shallow()
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    criterion = BCELoss()

    num_epochs = 1000

    progress_bar = tqdm(range(len(train_loader) * num_epochs))

    for epoch in range(num_epochs):
        for batch in train_loader:
            progress_bar.update(1)
            optim.zero_grad()
            input_ids = batch['input_ids'].type(torch.FloatTensor).to(device)
            labels = batch['labels'].type(torch.FloatTensor).to(device)
            outputs = torch.squeeze(model(input_ids))
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

    torch.save(
        {
            'model_state_dict':  model.state_dict(),
            'loss':              loss,
        },
        f'/home/gracen/repos/peace/models/shallow/trained_mod.pth'
    )
    
    model.eval()

    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    val_dataset = OneStopEnglishDataset(val_encodings, val_labels)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    progress_bar = tqdm(range(len(val_loader)))

    num_correct = 0
    num_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            progress_bar.update(1)
            input_ids = batch['input_ids'].type(torch.FloatTensor).to(device)
            labels = batch['labels'].type(torch.FloatTensor).to(device)
            outputs = torch.squeeze(model(input_ids))
            loss = criterion(outputs, labels)
            predictions = outputs.round()
            num_correct += torch.sum(predictions == labels)
            num_predictions += len(predictions)

    print('loss:', loss.item())
    print('accuracy:', (num_correct / num_predictions).item())


if __name__ == '__main__':
    main()