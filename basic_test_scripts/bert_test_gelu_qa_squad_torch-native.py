import torch, math, sys, os, argparse, evaluate, nltk, datetime, subprocess
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from pathlib import Path
from os.path import isdir
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertConfig
from models import BertForSequenceClassification, BertForMultipleChoice, BertForQuestionAnswering
from sacrebleu.metrics import BLEU, CHRF, TER


def main(gelu_type: str, home: str, checkpoint: str = None) -> None: # model_type: str, 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    config = BertConfig(
        hidden_dropout_prob=0, 
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096
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
        'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad'
        # pretrained,
    )
    model = BertForQuestionAnswering.from_pretrained(
        # 'bert-base-uncased',
        # "textattack/bert-base-uncased-yelp-polarity",
        'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad',
        # pretrained,
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)
    print(model)
    exit()

    # dataset = load_dataset('allenai/openbookqa')
    dataset = load_dataset("squad")
    ## ANSWER START - first character of the answer string in the un-tokenized context

    # print()
    # example = dataset['validation'][1]
    # inputs = tokenizer(example['question'], example['context'], return_tensors="pt")
    # for i, token in enumerate(inputs.input_ids[0]):
    #     print(i, ':', tokenizer.decode([token], skip_special_tokens=True))
    # print(example['answers']['answer_start'])
    # print(example['context'][example['answers']['answer_start'][0]:example['answers']['answer_start'][0] + len(example['answers']['text'][0]) + 1])

    # exit()
    
    batch_size = 1
    output_dir = f'{home}/qa/bert-large-uncased/{config.hidden_act}/test'

    # criterion = CrossEntropyLoss()
    # bleu = evaluate.load('bleu')
    sacre_bleu = BLEU(max_ngram_order=1) #, tokenize=tokenizer)

    ## keys - context, question, answers (text, answer_start)
    def preprocessing_function(examples):
        questions = [q.strip() for q in examples['question']]
        contexts = [c.strip() for c in examples['context']]
        # ans_texts = []
        ans_starts = []
        ans_ends = []
        for answer in examples['answers']:
            texts = answer['text']
            starts = answer['answer_start']
            ends = []
            for text, start in zip(texts, starts):
                ends.append(start + len(text))
            # ans_texts.append(texts)
            ans_starts.append(starts)
            ans_ends.append(ends)
        return {
            'questions': questions,
            'contexts': contexts,
            # 'ans_texts': ans_texts,
            'ans_starts': ans_starts,
            'ans_ends': ans_ends
        }


    test_dataset = dataset['validation'].map(
        preprocessing_function,
        batched=True,
        remove_columns=dataset['validation'].column_names,
    )
    test_dataloader = DataLoader(
        # dataset['validation'], 
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=1
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

    # total_loss = 0
    total_score = 0
    total_latency = 0
    total_power = 0
    total_throughput = 0
    iteration = 0
    model.eval()
    # criterion.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            # print(batch)
            questions, contexts, ans_starts, ans_ends = batch.values()
            # print(questions)
            # print(contexts)
            # print(ans_texts)
            # print(ans_starts)
            # print(ans_ends)
            # exit()
            # for start, end in zip(starts, ends):
            #     print(contexts[0][start:end])
            # exit()
            # question, context, ans_text, ans_start = batch['question'], batch['context'], batch['answers']['text'], batch['answers']['answer_start']
            inputs = tokenizer(questions, contexts, padding=True, truncation=True, return_tensors="pt").to(device)
            print(inputs.input_ids.size())
            exit()
            # print(question)
            # print(context)
            # print(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            # exit()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            end.record()
            torch.cuda.synchronize()           
            process = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout

            latency = start.elapsed_time(end) * 0.001
            power = float(process)
            throughput = inputs.input_ids.size(1) * batch_size * latency
            total_latency += latency
            total_power += power
            total_throughput += throughput
            ## convert answer start and ends from untokenized to tokenized versions
            # ans_starts_token = torch.zeros((len(ans_starts)))
            # ans_ends_token = torch.zeros((len(ans_ends)))
            # print(ans_starts)
            # print(ans_ends)
            # print(ans_starts_token)
            # print(ans_ends_token)
            # exit()
            # ignored_index = outputs.start_logits.shape[1]
            # start_positions = ans_starts_token.clamp(0, ignored_index)
            # end_positions = ans_ends_token.clamp(0, ignored_index)
            # criterion = CrossEntropyLoss(ignore_index=ignored_index)
            # start_loss = criterion(outputs.start_logits, start_positions)
            # end_loss = criterion(outputs.end_logits, end_positions)
            # total_loss += (start_loss + end_loss) / 2

            start_index = outputs.start_logits.argmax()
            end_index = outputs.end_logits.argmax()
            # print(start_index)
            # print(end_index)
            predict_answer_tokens = inputs.input_ids[0, start_index : end_index + 1]
            predict_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
            # print(contexts[0][ans_starts[0]:ans_ends[0]])
            # print(predict_answer)
            # ans_texts = []
            # for start, end in zip(ans_starts, ans_ends):
            #     ans_texts.append(contexts[0][start:end].lower().split())
            # ans_texts = list(set(ans_texts))
            # print(predict_answer.split())
            # print(ans_texts)
            # ans_texts = [ans_texts]
            # pred_texts = [predict_answer.split()]
            ans_texts = []
            for start, end in zip(ans_starts, ans_ends):
                ans_texts.append(contexts[0][start:end].lower())
            ans_texts = [ans_texts]
            pred_texts = [predict_answer]
            # print((ans_texts))
            # print((pred_texts))
            # score = nltk.translate.bleu_score.corpus_bleu(ans_texts, pred_texts, n=1)
            # results = bleu.compute(predictions=pred_texts, references=ans_texts, tokenizer=tokenizer, smooth=True)
            score = sacre_bleu.corpus_score(pred_texts, ans_texts).score
            # print(results)
            # print(score)
            # exit()
            # print(results)
            # exit()
            total_score += score

            iteration += 1

            if i > 0 and i % 100 == 0:
                with open(f'{output_dir}/testing_log.txt', 'a+') as f:
                    # f.write(f'testing loss & bleu score: {total_loss / iteration}, {total_score / iteration}\n')
                    f.write(f'testing bleu score, latency, power, throughput: {total_score / iteration}, {total_latency / iteration}, {total_power / iteration}, {total_throughput / iteration}\n')
                    # f.write(f'testing bleu score: {total_score / iteration}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type=str, help='size of bert model (base or large)', required=True)
    parser.add_argument('-g', '--gelu', type=str, help='gelu method (naive or approx)', required=True)
    parser.add_argument('-l', '--location', type=str, help='location of home directory for saving', required=True)
    # parser.add_argument('-c', '--checkpoint', type=str, help='previous training checkpoint to restart from')
    args = parser.parse_args()
    main(args.gelu, args.location) #args.model, , checkpoint=args.checkpoint)