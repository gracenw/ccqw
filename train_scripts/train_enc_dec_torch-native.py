# flake8: noqa

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForMultipleChoice, EncoderDecoderModel, GenerationConfig,AutoTokenizer, EncoderDecoderConfig
# from transformers.models.encoder_decoder.modeling_encoder_decoder EncoderDecoderForSeqClass

from custom_datasets.sentiment import SentimentDataset
from datasets import load_dataset

from pathlib import Path
import math
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW, SGD
from os.path import isdir
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


def main():
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    np.random.seed(42)

    learning_rate = 1e-3
    batch_size = 1
    weight_decay = 1e-4
    start_epoch = 0
    num_epochs = 1
    num_classes = 2
    checkpoint = ''
    output_dir = f'/home/gracen/repos/rampc/weights/summarize/bert'

    train_dataset = load_dataset("abisee/cnn_dailymail", '3.0.0', split="train")

    device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    config = EncoderDecoderConfig.from_encoder_decoder_configs(BertConfig(), BertConfig())
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "google-bert/bert-base-uncased", 
        "google-bert/bert-base-uncased",
        config=config
    )
    # model = nn.DataParallel(model)
    model.to(device)
    model.config.encoder._attn_implementation = 'sdpa'
    model.config.decoder._attn_implementation = 'sdpa'
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.decoder.config.decoder_start_token_id = tokenizer.cls_token_id,
    # model.decoder.config.eos_token_id = tokenizer.sep_token_id,
    # model.decoder.config.pad_token_id = tokenizer.pad_token_id,
    # model.decoder.config.vocab_size = model.config.encoder.vocab_size

    optimizer = SGD(
        params=model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    lr_scheduler = StepLR(
        optimizer=optimizer, 
        step_size=10
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    crtierion = CrossEntropyLoss()

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

    model.train()
    # criterion.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        iteration = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            articles, highlights, ids = batch['article'], batch['highlights'], batch['id']
            inputs = tokenizer(articles, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(device)
            input_mask = inputs.attention_mask.to(device)
            labels = tokenizer(highlights, return_tensors="pt", padding=True, truncation=True)
            label_ids = labels.input_ids.to(device)
            # label_mask = labels.attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
            # exit()

            # loss = outputs.loss
            loss = crtierion(outputs.logits.reshape(-1, model.decoder.config.vocab_size), labels.view(-1))
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
            

        lr_scheduler.step(epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, Path(f'{output_dir}/checkpoint.pth'))


if __name__ == '__main__':
    # exit()

    # config_encoder = BertConfig(
    #     # hidden_dropout_prob=0, 
    #     _attn_implementation = 'sdpa',
    # )
    # config_decoder = BertConfig(
    #     # hidden_dropout_prob=0, 
    #     _attn_implementation = 'sdpa',
    #     is_decoder=True,
    #     add_cross_attention=True
    # )
    # config = EncoderDecoderConfig.from_encoder_decoder_configs(
    #     config_encoder, 
    #     config_decoder,
    #     decoder_start_token_id = tokenizer.cls_token_id,
    #     eos_token_id = tokenizer.sep_token_id,
    #     pad_token_id = tokenizer.pad_token_id,
    #     vocab_size = config_encoder.vocab_size
    # )
    # model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail", config=config)
    # print(model.config._attn_implementation)
    
    # ARTICLE_TO_SUMMARIZE = (
    #     "The cat (Felis catus), also referred to as the domestic cat, is a small domesticated carnivorous mammal. "
    #     "It is the only domesticated species of the family Felidae. Advances in archaeology and genetics have shown "
    #     "that the domestication of the cat occurred in the Near East around 7500 BC. It is commonly kept as a pet "
    #     "and farm cat, but also ranges freely as a feral cat avoiding human contact. It is valued by humans for "
    #     "companionship and its ability to kill vermin. Its retractable claws are adapted to killing small prey "
    #     "species such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth, and its "
    #     "night vision and sense of smell are well developed. It is a social species, but a solitary hunter and a "
    #     "crepuscular predator. Cat communication includes vocalizations—including meowing, purring, trilling, "
    #     "hissing, growling, and grunting—as well as body language. It can hear sounds too faint or too high in "
    #     "frequency for human ears, such as those made by small mammals. It secretes and perceives pheromones."
    # )

    # input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids
    # outputs = model(input_ids=input_ids, labels=labels)
    # loss, logits = outputs.loss, outputs.logits
    # print(loss)

    # gen_config = GenerationConfig(
    #     decoder_start_token_id = tokenizer.cls_token_id,
    #     max_length=50,
    # )
    # generated_ids = model.generate(input_ids, generation_config=gen_config)
    # generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    main()