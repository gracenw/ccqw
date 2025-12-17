import torch, math, sys, os, argparse, evaluate, nltk, datetime
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from pathlib import Path
from os.path import isdir
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertConfig, EncoderDecoderConfig
from models import BertForSequenceClassification, BertForMultipleChoice, BertForQuestionAnswering, EncoderDecoderModel
from sacrebleu.metrics import BLEU, CHRF, TER


def main(home: str, device: str, checkpoint: str = None) -> None: # model_type: str,
    if not (device  == 'cuda' or device == 'cpu'):
        print('please enter a valid device type: cuda or cpu')
        exit()

    device = torch.device(device)
    # device = torch.device("cuda") if device == 'cuda' else torch.device("cpu")
    
    tokenizer = BertTokenizerFast.from_pretrained(
        "patrickvonplaten/bert2bert_cnn_daily_mail",
    )

    # config = EncoderDecoderConfig.from_encoder_decoder_configs(
    #     BertConfig(
    #         hidden_dropout_prob=0,
    #         hidden_act=hidden_act,
    #     ),
    #     BertConfig(
    #         hidden_dropout_prob=0,
    #         hidden_act=hidden_act,
    #         decoder_start_token_id = tokenizer.cls_token_id,
    #         pad_token_id = tokenizer.pad_token_id,
    #     )
    # )
    # config.max_length = 142

    model = EncoderDecoderModel.from_pretrained(
        "patrickvonplaten/bert2bert_cnn_daily_mail",
        # config=config,
    ).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

    dataset = load_dataset("abisee/cnn_dailymail", '3.0.0')

    # print(dataset['test'][0].keys())
    # exit()
    
    batch_size = 1
    output_dir = f'{home}/summary/bert-seq2seq-base/embedding/test'

    # sacre_bleu = BLEU(max_ngram_order=1) #, tokenize=tokenizer) 

    test_dataloader = DataLoader(
        dataset['test'], 
        batch_size=batch_size,
        shuffle=True,
    )

    # output_dir = Path(output_dir)
    # if not isdir(output_dir):
    #     Path(output_dir).mkdir(parents=True, exist_ok=True) 
    # last_version = -1
    # for path in os.listdir(output_dir):
    #     if 'version_' in path:
    #         version_num = (int) (path.split('_')[-1])
    #         last_version = version_num if version_num > last_version else last_version
    # output_dir = Path(f'{output_dir}/version_{last_version + 1:03}')
    # Path(output_dir).mkdir(parents=True, exist_ok=True) 

    # with open(f'{output_dir}/testing_log.txt', 'a+') as f:
    #     f.write(f'Test run beginning on {datetime.datetime.now()}, measuring the power (W), latency (s), and throughput (seq/s) of embedding\n')
    
    # if os.path.exists('scratchpad.txt'):
    #     os.remove('scratchpad.txt')

    total_power = 0
    total_latency = 0
    total_throughput = 0
    iteration = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            articles, summaries = batch['article'], batch['highlights']
            inputs = tokenizer(articles, padding=True, truncation=True, return_tensors="pt").to(device)
            generated_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
            # generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # score = sacre_bleu.corpus_score([generated_text], [summaries]).score
            # total_score += score

            iteration += 1
            if iteration == 5:
                break

            # if i > 0 and i % 100 == 0:
            #     with open(f'{output_dir}/testing_log.txt', 'a+') as f:
            #         f.write(f'testing embedding power, latency, throughput: \n')

    with open('scratchpad.txt') as f:
        lines = []
        for line in f:
            lines.append([float(entry) for entry in line.rstrip().split('-')])
    os.remove('scratchpad.txt')
            
    metrics = torch.tensor(lines, device=device, dtype=torch.float)
    averages = metrics.mean(dim=0)

    print('bert embedding metrics')
    print(f'average power consumption: {averages[0]} W')
    print(f'average latency: {averages[1]} s')
    print(f'average throughput: {averages[2]} seq/s')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, help='device type to run model', required=True)
    parser.add_argument('-l', '--location', type=str, help='location of home directory for saving', required=True)
    args = parser.parse_args()
    main(args.location, args.device)