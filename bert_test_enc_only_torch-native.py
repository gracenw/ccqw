# flake8: noqa

from torch import nn
import torch
import numpy as np
from transformers import BertTokenizerFast, BertConfig

from models import BertForSequenceClassification, BertForMultipleChoice


def main():
    # device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    config = BertConfig(
        hidden_dropout_prob=0, 
        _attn_implementation = 'sdpa',
        hidden_act='naive_gelu'
    )
    num_to_label = ['negative', 'positive']
    tokenizer = BertTokenizerFast.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity"
        # 'bert-base-uncased'
    )
    # checkpoint = '/home/gracen/repos/rampc/models/weights/polarity/bert-base-uncased/naive_gelu/version_000/checkpoint.pth'
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity", 
        # 'bert-base-uncased',
        config=config
    ).to(device)
    # checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # model.load_state_dict(checkpoint['model'])

    sentence = "my cat is the cutest cat ever!"
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, labels=torch.tensor([1]))
    predict = outputs.logits.argmax().item()
    loss = outputs.loss
    print(f'{sentence} --> {num_to_label[predict]}')
    print(f'loss: {loss}')

    sentence = "i do not like running outside in the wind and cold."
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=torch.tensor([0]))
    predict = outputs.logits.argmax().item()
    loss = outputs.loss
    print(f'{sentence} --> {num_to_label[predict]}')
    print(f'loss: {loss}')


if __name__ == '__main__':
    main()