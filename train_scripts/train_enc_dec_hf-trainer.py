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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW, SGD
from os.path import isdir
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from datasets import concatenate_datasets, DatasetDict


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

    dataset = load_dataset("abisee/cnn_dailymail", '3.0.0')
    # train_dataset = dataset['train']
    # val_dataset = dataset['validation']

    device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
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

    max_input_length = 512
    max_target_length = 30


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

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    rouge_score = evaluate.load('rouge')

    # generated_summary = "I absolutely loved reading the Hunger Games"
    # reference_summary = "I loved reading the Hunger Games"

    # scores = rouge_score.compute(
    #     predictions=[generated_summary], references=[reference_summary]
    # )

    # Show the training loss with every epoch
    logging_steps = len(tokenized_dataset["train"]) // batch_size

    import accelerate
    print(accelerate.__version__)

    args = Seq2SeqTrainingArguments(
        output_dir='bert2bert_articles',
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=False,
    )

    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download("punkt")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
    
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    tokenized_dataset = tokenized_dataset.remove_columns(
        dataset["train"].column_names
    )

    features = [tokenized_dataset["train"][i] for i in range(2)]
    data_collator(features)

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    


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