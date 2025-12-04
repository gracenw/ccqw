import torch
from tqdm import tqdm
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from data import read_test_data, OneStopEnglishDataset


def main():
    test_texts, test_labels = read_test_data('/home/gracen/repos/peace/data')

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    test_dataset = OneStopEnglishDataset(test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DistilBertForSequenceClassification.from_pretrained('/home/gracen/repos/peace/models/distilbert')
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    progress_bar = tqdm(range(len(test_loader)))

    num_correct = 0
    num_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
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