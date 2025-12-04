# flake8: noqa
import torch, pathlib, datasets, evaluate, os, tqdm, math, sys
import numpy as np
from models import T5ForConditionalGeneration
from transformers import T5Tokenizer#, T5ForConditionalGeneration


def main():
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    np.random.seed(42)

    learning_rate = 2e-5
    weight_decay = 1e-2
    batch_size = 1
    start_epoch = 0
    num_epochs = 10
    output_dir = '/home/gracen/repos/rampc/models/weights/t5_absolute'
    device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=False)
    books = datasets.load_dataset("opus_books", "de-en")
    books = books["train"].train_test_split(test_size=0.2)

    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    checkpoint = '/home/gracen/repos/rampc/models/weights/t5_absolute/version_001'

    if not os.path.isdir(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
    last_version = -1
    for path in os.listdir(output_dir):
        if 'version_' in path:
            version_num = (int) (path.split('_')[-1])
            last_version = version_num if version_num > last_version else last_version
    output_dir = pathlib.Path(f'{output_dir}/version_{last_version + 1:03}')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 

    if checkpoint:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    else:
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
        enc_pos_emb = np.load('/home/gracen/repos/rampc/models/kernels/enc_pos_emb.npy')
        dec_pos_emb = np.load('/home/gracen/repos/rampc/models/kernels/dec_pos_emb.npy')
        model.encoder.position_tokens.weight = torch.nn.Parameter(torch.tensor(enc_pos_emb))
        model.decoder.position_tokens.weight = torch.nn.Parameter(torch.tensor(dec_pos_emb))

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        iteration = 0
        for i, batch in enumerate(tqdm.tqdm(books['train'])):
            source = ['Translate English to German: ' + batch['translation']['en']]
            target = [batch['translation']['de']]
            inputs = tokenizer(source, text_target=target, return_tensors="pt", max_length=128, truncation=True, padding=True)

            input_ids = inputs['input_ids'].to(device)
            mask = inputs['attention_mask'].to(device)
            labels = inputs['labels'].to(device)       
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)

            loss_value = outputs.loss.item()
            total_loss += loss_value
            iteration += 1

            if not math.isfinite(loss_value):
                print(f'loss is infinite, stopping training')
                sys.exit(1)
            
            outputs.loss.backward()
            if i > 0:
                if i % 10 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if i % 100 == 0:
                        with open(f'{output_dir}/training_log.txt', 'a+') as f:
                            f.write(f'training loss: {total_loss / iteration}\n')
                        if i % 1000 == 0:
                            model.save_pretrained(output_dir)

        model.save_pretrained(output_dir)


if __name__ == '__main__':
    main()