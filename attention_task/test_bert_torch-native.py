# flake8: noqa

from torch import nn
import torch
import numpy as np
from transformers import BertTokenizerFast

from ..models.modeling_bert_local import BertConfig, BertForSequenceClassification, BertForMultipleChoice


def main():
    # device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    config = BertConfig(
        hidden_dropout_prob=0, 
        _attn_implementation = 'sdpa',
    )
    num_to_label = ['negative', 'positive']
    tokenizer = BertTokenizerFast.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity"
    )
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity", 
        config=config
    ).to(device)

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