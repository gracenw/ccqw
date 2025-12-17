# flake8: noqa
import torch, evaluate
import numpy as np
from models import T5ForConditionalGeneration
from transformers import T5Tokenizer #, T5ForConditionalGeneration



def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def main():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('/home/gracen/repos/rampc/models/weights/t5_absolute/version_001')
    model.eval()

    input_ids = tokenizer(
        "translate English to German: I am moving out of my apartment next week.", return_tensors="pt"
    ).input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    metric = evaluate.load("sacrebleu")

    target_ids = tokenizer(
        'ich liebe meine Katze sehr', return_tensors="pt"
    ).input_ids

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple): preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     decoded_preds = [pred.strip() for pred in decoded_preds]
    #     decoded_labels = [[label.strip()] for label in decoded_labels]
    #     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    #     result = {"bleu": result["score"]}
    #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     result = {k: round(v, 4) for k, v in result.items()}
    #     return result
        
    # print(compute_metrics((outputs[0], target_ids)))

    # model.save_pretrained("/home/gracen/repos/rampc/models/weights/t5_absolute", from_pt=True) 
    # import numpy as np
    # enc_pos_emb = np.load('/home/gracen/repos/rampc/models/kernels/enc_pos_emb.npy')
    # dec_pos_emb = np.load('/home/gracen/repos/rampc/models/kernels/dec_pos_emb.npy')
    # model.encoder.position_tokens.weight = torch.nn.Parameter(torch.tensor(enc_pos_emb))
    # model.decoder.position_tokens.weight = torch.nn.Parameter(torch.tensor(dec_pos_emb))
    # print(model.config)
    # input_ids = tokenizer(
    #     "translate English to German: I love my cat very much!", return_tensors="pt"
    # ).input_ids
    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # from transformers import T5ForConditionalGeneration
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    # disable_dropout(model)
    # input_ids = tokenizer(
    #     "translate English to Spanish: I love my cat very much!", return_tensors="pt"
    # ).input_ids
    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # checkpoint = torch.load('/home/gracen/repos/rampc/lightning_logs/version_1/checkpoints/check.ckpt', map_location='cpu')
    # state_dict = {}
    # for key, tensor in checkpoint['state_dict'].items():
    #     state_dict[key.replace('model.', '')] = tensor
    # model = T5ForConditionalGeneration(config=base_model.config)
    # model.load_state_dict(state_dict)
    # model.to('cpu')
    # model.eval()
    # input_ids = tokenizer(
    #     "translate English to German: Fridays are my favorite days.", return_tensors="pt"
    # ).input_ids
    # outputs = model.generate(input_ids)
    # print(outputs)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # print(outputs)
    # print(tokenizer.decode(torch.tensor([13959,  1566,    12,  2968]).unsqueeze(0), skip_special_tokens=True))
    # print(model)
    # article = (
    #     "summarize: PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    #     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    #     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    # )
    # tokenized = tokenizer(article, return_tensors="pt")
    # input_ids = tokenized.input_ids
    # input_mask = tokenized.attention_mask
    # generated_ids = model.generate(input_ids, attention_mask=input_mask)
    # generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(generated_text)


if __name__ == '__main__':
    main()