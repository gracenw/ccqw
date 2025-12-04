import torch, math, sys, os, argparse, datetime, time, subprocess
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from pathlib import Path
from os.path import isdir
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertConfig
from models import BertForSequenceClassification, BertForMultipleChoice


def main(gelu_type: str, home: str, checkpoint: str = None) -> None: # model_type: str, 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    config = BertConfig(
        hidden_dropout_prob=0, 
    )

    if gelu_type == 'approx':
        config.hidden_act = 'approx_gelu'
    elif gelu_type == 'naive':
        config.hidden_act = 'naive_gelu'
    else:
        print('please enter a valid gelu type: naive or approx')
        exit()

    # if model_type == 'base':
    #     # model_type = 'bert-base-uncased'
    #     pretrained = "textattack/bert-base-uncased-yelp-polarity"
    #     dataset = load_dataset("imdb")
    # elif model_type == 'large':
    #     # model_type == 'bert-large-uncased'
    #     pretrained == 'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad'
    #     dataset = load_dataset('allenai/openbookqa')
    # else:
    #     print('please enter a valid model size: base or large')
    #     exit()

    tokenizer = BertTokenizerFast.from_pretrained(
        # 'bert-base-uncased',
        "textattack/bert-base-uncased-yelp-polarity",
        # pretrained,
    )
    model = BertForSequenceClassification.from_pretrained(
        # 'bert-base-uncased',
        "textattack/bert-base-uncased-yelp-polarity",
        # pretrained,
        config=config,
    ).to(device)

    # dataset = load_dataset("imdb")
    dataset = load_dataset('yelp_review_full')
    # print(dataset['test'][0])
    # exit()

    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    batch_size = 1
    output_dir = f'{home}/polarity/bert-base-uncased/{config.hidden_act}/test'

    criterion = CrossEntropyLoss()

    # tokenized_test_dataset = dataset['test'].map(
    #     lambda examples: tokenizer(
    #         examples['text'], 
    #         padding=True, 
    #         truncation=True
    #     ), 
    #     batched=True
    # )
    # tokenized_test_dataset.set_format(
    #     type='torch', 
    #     columns=['input_ids', 'token_type_ids', 'attention_mask', 'label']
    # )
    test_dataloader = DataLoader(
        # tokenized_test_dataset, 
        dataset['test'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    output_dir = Path(output_dir)
    if not isdir(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True) 
    last_version = -1
    for path in os.listdir(output_dir):
        if 'version_' in path:
            version_num = (int) (path.split('_')[-1])
            last_version = version_num if version_num > last_version else last_version
    output_dir = Path(f'{output_dir}/version_{last_version + 1:03}')
    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    with open(f'{output_dir}/testing_log.txt', 'a+') as f:
        f.write(f'Test run beginning on {datetime.datetime.now()}, measuring accuracy (%), latency (s), power (W), throughput (seq/s)\n')

    total_loss = 0
    total_accuracy = 0
    total_latency = 0
    total_power = 0
    total_throughput = 0
    iteration = 0
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            # print(batch['label'].dtype)
            # input_ids, input_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            labels, texts = (batch['label'] > 2).int().to(device), batch['text']
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            # print(inputs.input_ids.size())
            # exit()
            ## we are gonna say 0-2 is negative, 3-4 is positive

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)            
            end.record()
            torch.cuda.synchronize()           
            process = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout

            latency = start.elapsed_time(end) * 0.001
            power = float(process)
            throughput = inputs.input_ids.size(1) * batch_size * latency
            total_latency += latency
            total_power += power
            total_throughput += throughput

            logits = output.logits

            # loss = criterion(logits, labels)
            # loss_value = loss.item()
            # total_loss += loss_value

            predicted = logits.max(1).indices
            accuracy = (predicted == labels).sum() / predicted.shape[0]
            total_accuracy += accuracy.item()

            iteration += 1

            if i > 0 and i % 100 == 0:
                with open(f'{output_dir}/testing_log.txt', 'a+') as f:
                    # f.write(f'testing loss & accuracy: {total_loss / iteration}, {total_accuracy / iteration}\n')
                    f.write(f'testing accuracy, latency, power, throughput: {total_accuracy / iteration}, {total_latency / iteration}, {total_power / iteration}, {total_throughput / iteration}\n')
                    # exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type=str, help='size of bert model (base or large)', required=True)
    parser.add_argument('-g', '--gelu', type=str, help='gelu method (naive or approx)', required=True)
    parser.add_argument('-l', '--location', type=str, help='location of home directory for saving', required=True)
    # parser.add_argument('-c', '--checkpoint', type=str, help='previous training checkpoint to restart from')
    args = parser.parse_args()
    main(args.gelu, args.location) #args.model, , checkpoint=args.checkpoint)