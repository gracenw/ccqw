import torch, os
from tqdm import tqdm
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from data import read_train_data, OneStopEnglishDataset


def main():
    texts, labels = read_train_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)

    train_dataset = OneStopEnglishDataset(train_encodings, train_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):
        progress_bar = tqdm(range(len(train_loader)))
        for batch in train_loader:
            progress_bar.update(1)
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.save_pretrained(f'{os.getcwd()}/models/distilbert', from_pt=True)
    
    model.eval()

    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    val_dataset = OneStopEnglishDataset(val_encodings, train_labels)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    progress_bar = tqdm(range(len(val_loader)))

    num_correct = 0
    num_predictions = 0

    with torch.no_grad():
        for batch in val_loader:
            progress_bar.update(1)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = Softmax(dim=1)(outputs.logits).argmax(-1)
            num_correct += torch.sum(predictions == labels)
            num_predictions += len(predictions)

    print('loss:', outputs[0].item())
    print('accuracy:', (num_correct / num_predictions).item())


if __name__ == '__main__':
    main()