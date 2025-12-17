# flake8: noqa
import torch, pathlib, datasets, evaluate
import numpy as np
from models import T5ForConditionalGeneration
from transformers import T5Tokenizer, DataCollatorForSeq2Seq #, T5ForConditionalGeneration


def main():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    # for name, param in model.named_parameters():
    #     if not 'position_embeddings' in name:
    #         param.requries_grad = False

    enc_pos_emb = np.load('/home/gracen/repos/rampc/models/kernels/enc_pos_emb.npy')
    dec_pos_emb = np.load('/home/gracen/repos/rampc/models/kernels/dec_pos_emb.npy')
    model.encoder.position_embeddings.weight = torch.nn.Parameter(torch.tensor(enc_pos_emb))
    model.decoder.position_embeddings.weight = torch.nn.Parameter(torch.tensor(dec_pos_emb))

    # input_ids = tokenizer(
    #     "translate English to German: I love my cat very much!", return_tensors="pt"
    # ).input_ids
    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    books = datasets.load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)
    source_lang = "en"
    target_lang = "fr"
    prefix = "translate English to French: "

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets) #, max_length=128, truncation=True)
        return model_inputs
    
    tokenized_books = books.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    metric = evaluate.load("sacrebleu")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="t5_pos_emb_first",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True, #change to bf16=True for XPU
        # push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.01),
    )

    trainer.train()


if __name__ == '__main__':
    main()