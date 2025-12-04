import torch, math, sys, os
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from pathlib import Path
from os.path import isdir
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding

from ..models.modeling_bert_local import BertConfig, BertForSequenceClassification, BertForMultipleChoice


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = load_dataset("imdb")
    config = BertConfig(
        hidden_dropout_prob=0, 
        hidden_act='gelu'
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        'bert-base-uncased',
        # "textattack/bert-base-uncased-yelp-polarity",
    )
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        # "textattack/bert-base-uncased-yelp-polarity",
        config=config,
    ).to(device)

    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    learning_rate = 2e-4
    batch_size = 1
    weight_decay = 1e-2
    start_epoch = 0
    num_epochs = 10
    num_classes = 2
    checkpoint = ''
    output_dir = f'/home/gracen/repos/rampc/weights/sentiment/bert-base/gelu'

    optimizer = SGD(
        params=model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    lr_scheduler = StepLR(
        optimizer=optimizer, 
        step_size=10
    )
    criterion = CrossEntropyLoss()

    train_dataloader = DataLoader(
        dataset['train'], 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_dataloader = DataLoader(
        dataset['test'], 
        batch_size=batch_size,
        shuffle=True,
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
            Path(output_dir).mkdir(parents=True, exist_ok=True) 
        last_version = -1
        for path in os.listdir(output_dir):
            if 'version_' in path:
                version_num = (int) (path.split('_')[-1])
                last_version = version_num if version_num > last_version else last_version
        output_dir = Path(f'{output_dir}/version_{last_version + 1:03}')
        Path(output_dir).mkdir(parents=True, exist_ok=True) 

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        iteration = 0
        model.train()
        criterion.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            text, labels = batch['text'], batch['label']
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(device)
            input_mask = inputs.attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=input_mask).logits

            loss = criterion(logits, labels)
            loss_value = loss.item()
            total_loss += loss_value
            iteration += 1

            if not math.isfinite(loss_value):
                print(f'loss is infinite, stopping training')
                sys.exit(1)
            
            loss.backward()
            if i > 0:
                if i % 10 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if i % 100 == 0:
                        with open(f'{output_dir}/training_log.txt', 'a+') as f:
                            f.write(f'training loss: {total_loss / iteration}\n')
                        if i % 1000 == 0:
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                            }, Path(f'{output_dir}/checkpoint.pth'))

        # lr_scheduler.step(epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, Path(f'{output_dir}/checkpoint.pth'))

        total_loss = 0
        total_accuracy = 0
        iteration = 0
        model.eval()
        criterion.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader)):
                text, labels = batch['text'], batch['label']
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs.input_ids.to(device)
                input_mask = inputs.attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_ids=input_ids, attention_mask=input_mask).logits

                loss = criterion(logits, labels)
                loss_value = loss.item()
                total_loss += loss_value

                predicted = logits.max(1).indices
                accuracy = (predicted == labels).sum() / predicted.shape[0]
                accuracy_value = accuracy.item()
                total_accuracy += accuracy_value

                iteration += 1

                if i > 0 and i % 100 == 0:
                    with open(f'{output_dir}/testing_log.txt', 'a+') as f:
                        f.write(f'testing loss & accuracy: {total_loss / iteration}, {total_accuracy / iteration}\n')


if __name__ == '__main__':
    main()