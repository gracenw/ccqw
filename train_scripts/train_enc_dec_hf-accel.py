# flake8: noqa

from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification, BertForMultipleChoice, EncoderDecoderModel, GenerationConfig,AutoTokenizer, EncoderDecoderConfig
# from transformers.models.encoder_decoder.modeling_encoder_decoder EncoderDecoderForSeqClass
from transformers import Seq2SeqTrainingArguments

from custom_datasets.sentiment import SentimentDataset
from datasets import load_dataset
import evaluate
from pathlib import Path
import math
import sys
import os
import numpy as np
import torch
from torch import nn
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW, SGD
from os.path import isdir
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from datasets import concatenate_datasets, DatasetDict
from torch.optim import SGD

import nltk
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from accelerate import Accelerator

from tqdm.auto import tqdm
import torch
import numpy as np


nltk.download("punkt")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 1
    output_dir = f'/home/gracen/repos/rampc/bert2bert_accel'

    datasets = load_dataset("abisee/cnn_dailymail", '3.0.0')

    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    config = EncoderDecoderConfig.from_encoder_decoder_configs(BertConfig(), BertConfig())
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "google-bert/bert-base-uncased", 
        "google-bert/bert-base-uncased",
        config=config
    )
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    # model = EncoderDecoderModel.from_pretrained(output_dir)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["article"],
            max_length=512,
            truncation=True,
        )
        labels = tokenizer(
            examples["highlights"], max_length=25, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(datasets["train"].column_names) 
    tokenized_datasets.set_format("torch")

    rouge_score = evaluate.load('rouge')

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        collate_fn=data_collator, 
        batch_size=batch_size
    )
    
    optimizer = SGD(model.parameters(), lr=2e-5)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels
    
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if step % 1000 == 0:
                print(f'loss: {loss.item()}')

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                gen_config = GenerationConfig(
                    pad_token_id = 0,
                    decoder_start_token_id = 101,
                )
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    generation_config=gen_config,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        result = rouge_score.compute()
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch}:", result)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)


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