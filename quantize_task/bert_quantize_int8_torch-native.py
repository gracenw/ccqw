# flake8: noqa

from transformers import BertTokenizerFast, BertConfig

from models import BertForSequenceClassification, BertForMultipleChoice
from datasets import load_dataset
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver

import matplotlib.pyplot as plt

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(42)


def main():
    # device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    # config = BertConfig(
    #     hidden_dropout_prob=0, 
    #     _attn_implementation = 'sdpa',
    #     hidden_act='naive_gelu'
    # )
    num_to_label = ['negative', 'positive']
    tokenizer = BertTokenizerFast.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity"
        # 'bert-base-uncased'
    )
    # checkpoint = '/home/gracen/repos/rampc/models/weights/polarity/bert-base-uncased/naive_gelu/version_000/checkpoint.pth'
    model_fp32 = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity", 
        # 'bert-base-uncased',
        # config=config
    ).to(device)
    # checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # model.load_state_dict(checkpoint['model'])

    for module in model_fp32.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

    sentence = "my cat is the cutest cat ever!"
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # print(model_fp32)
    # exit()

    # def register_hooks(model):
    #     activations = {}
    #     def get_activation(name):
    #         def hook(model, input, output):
    #             activations[name] = {'input': input, 'output': output}
    #         return hook
    #     for name, module in model.named_modules():
    #         module.register_forward_hook(get_activation(name))
    #     return activations
    # activations = register_hooks(model_fp32)

    # with torch.no_grad():
    #     model_fp32(**inputs)

    # print(activations[f"bert.encoder.layer.0.attention.self"]["input"][0].dtype)

    # print(min(inputs), max(inputs))
    # print(inputs)

    # for name, param in model_fp32.named_parameters():
    #     print(name, torch.min(param), torch.max(param))

    # embedding_i = activations['bert.embeddings']['input']
    # print(embedding_i)
    # exit()
    # print(torch.aminmax(embedding_i))
    # embedding_o = activations['bert.embeddings']['output']
    # print(torch.aminmax(embedding_o))

    def plot_schemes(tensor, name):
        tensor = tensor.flatten().detach().numpy()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        sym_range = -max(max_val, abs(min_val)), max(max_val, abs(min_val))
        aff_range = min_val, max_val

        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(6)
        fig.set_figwidth(14)

        axs[0].set_title("affine")
        aff_plt, _, _ = axs[0].hist(tensor, density=True, bins=100)
        ymin, ymax = np.quantile(aff_plt[aff_plt>0], [0.25, 0.95])
        axs[0].vlines(x=aff_range, ls='--', colors='purple', ymin=ymin, ymax=ymax)

        axs[1].set_title("symmetric")
        sym_plt, _, _ = axs[1].hist(tensor, density=True, bins=100)
        ymin, ymax = np.quantile(sym_plt[sym_plt>0], [0.25, 0.95])
        axs[1].vlines(x=sym_range, ls='--', colors='purple', ymin=ymin, ymax=ymax)

        plt.savefig(f'/home/gracen/repos/rampc/quant/hist/{name}.png')
        plt.clf()
        plt.close()

    # for l in range(12):
    #     block = model_fp32.bert.encoder.layer[l]
    #     prefix = 'bert.encoder'

    #     attn_i = activations[f"{prefix}.layer.{l}.attention.self"]["input"][0]
    #     plot_schemes(attn_i, f'attn_i_{l}')
    #     # print(torch.aminmax(attn_i))
    #     query_w = block.attention.self.query.weight
    #     plot_schemes(query_w, f'query_w_{l}')
    #     # print(torch.aminmax(query_w))
    #     query_b = block.attention.self.query.bias
    #     plot_schemes(query_b, f'query_b_{l}')
    #     # print(torch.aminmax(query_b))
    #     query_o = activations[f"{prefix}.layer.{l}.attention.self.query"]["output"][0]
    #     plot_schemes(query_o, f'query_o_{l}')
    #     # print(torch.aminmax(query_o))
    #     key_w = block.attention.self.key.weight
    #     plot_schemes(key_w, f'key_w_{l}')
    #     # print(torch.aminmax(key_w))
    #     key_b = block.attention.self.key.bias
    #     plot_schemes(key_b, f'key_b_{l}')
    #     # print(torch.aminmax(key_b))
    #     key_o = activations[f"{prefix}.layer.{l}.attention.self.key"]["output"][0]
    #     plot_schemes(key_o, f'key_o_{l}')
    #     # print(torch.aminmax(key_o))
    #     value_w = block.attention.self.value.weight
    #     plot_schemes(value_w, f'value_w_{l}')
    #     # print(torch.aminmax(value_w))
    #     value_b = block.attention.self.value.bias
    #     plot_schemes(value_b, f'value_b_{l}')
    #     # print(torch.aminmax(value_b))
    #     value_o = activations[f"{prefix}.layer.{l}.attention.self.value"]["output"][0]
    #     plot_schemes(value_o, f'value_o_{l}')
    #     # print(torch.aminmax(value_o))
    #     attn_o = activations[f"{prefix}.layer.{l}.attention.self"]["output"][0]
    #     plot_schemes(attn_o, f'attn_o_{l}')
    #     # print(torch.aminmax(attn_o))

    #     dense_i = activations[f"{prefix}.layer.{l}.attention.output.dense"]["input"][0]
    #     plot_schemes(dense_i, f'dense_i_{l}')
    #     # print(torch.aminmax(dense_i))
    #     dense_w = block.attention.output.dense.weight
    #     plot_schemes(dense_w, f'dense_w_{l}')
    #     # print(torch.aminmax(dense_w))
    #     dense_b = block.attention.output.dense.bias
    #     plot_schemes(dense_b, f'dense_b_{l}')
    #     # print(torch.aminmax(dense_b))
    #     dense_o = activations[f"{prefix}.layer.{l}.attention.output.dense"]["output"][0]
    #     plot_schemes(dense_o, f'dense_o_{l}')
    #     # print(torch.aminmax(dense_o))
        
    #     layernorm1_i = activations[f"{prefix}.layer.{l}.attention.output.LayerNorm"]["input"][0]
    #     plot_schemes(layernorm1_i, f'layernorm1_i_{l}')
    #     # print(torch.aminmax(layernorm1_i))
    #     layernorm1_w = block.attention.output.LayerNorm.weight
    #     plot_schemes(layernorm1_w, f'layernorm1_w_{l}')
    #     # print(torch.aminmax(layernorm1_w))
    #     layernorm1_b = block.attention.output.LayerNorm.bias
    #     plot_schemes(layernorm1_b, f'layernorm1_b_{l}')
    #     # print(torch.aminmax(layernorm1_b))
    #     layernorm1_o = activations[f"{prefix}.layer.{l}.attention.output.LayerNorm"]["output"][0]
    #     plot_schemes(layernorm1_o, f'layernorm1_o_{l}')
    #     # print(torch.aminmax(layernorm1_o))

    #     ffn1_i = activations[f"{prefix}.layer.{l}.intermediate.dense"]["input"][0]
    #     plot_schemes(ffn1_i, f'ffn1_i_{l}')
    #     # print(torch.aminmax(ffn1_i))
    #     ffn1_w = block.intermediate.dense.weight
    #     plot_schemes(ffn1_w, f'ffn1_w_{l}')
    #     # print(torch.aminmax(ffn1_w))
    #     ffn1_b = block.intermediate.dense.bias
    #     plot_schemes(ffn1_b, f'ffn1_b_{l}')
    #     # print(torch.aminmax(ffn1_b))
    #     ffn1_o = activations[f"{prefix}.layer.{l}.intermediate.dense"]["output"][0]
    #     plot_schemes(ffn1_o, f'ffn1_o_{l}')
    #     # print(torch.aminmax(ffn1_o))
        
    #     gelu_i = activations[f"{prefix}.layer.{l}.intermediate.intermediate_act_fn"]["input"][0]
    #     plot_schemes(gelu_i, f'gelu_i_{l}')
    #     # print(torch.aminmax(gelu_i))
    #     gelu_o = activations[f"{prefix}.layer.{l}.intermediate.intermediate_act_fn"]["output"][0]
    #     plot_schemes(gelu_o, f'gelu_o_{l}')
    #     # print(torch.aminmax(gelu_o))

    #     ffn2_i = activations[f"{prefix}.layer.{l}.output.dense"]["input"][0]
    #     plot_schemes(ffn2_i, f'ffn2_i_{l}')
    #     # print(torch.aminmax(ffn2_i))
    #     ffn2_w = block.output.dense.weight
    #     plot_schemes(ffn2_w, f'ffn2_w_{l}')
    #     # print(torch.aminmax(ffn2_w))
    #     ffn2_b = block.output.dense.bias
    #     plot_schemes(ffn2_b, f'ffn2_b_{l}')
    #     # print(torch.aminmax(ffn2_b))
    #     ffn2_o = activations[f"{prefix}.layer.{l}.output.dense"]["output"][0]
    #     plot_schemes(ffn2_o, f'ffn2_o_{l}')
    #     # print(torch.aminmax(ffn2_o))
        
    #     layernorm2_i = activations[f"{prefix}.layer.{l}.output.LayerNorm"]["input"][0]
    #     plot_schemes(layernorm2_i, f'layernorm2_i_{l}')
    #     # print(torch.aminmax(layernorm2_i))
    #     layernorm2_w = block.output.LayerNorm.weight
    #     plot_schemes(layernorm2_w, f'layernorm2_w_{l}')
    #     # print(torch.aminmax(layernorm2_w))
    #     layernorm2_b = block.output.LayerNorm.bias
    #     plot_schemes(layernorm2_b, f'layernorm2_b_{l}')
    #     # print(torch.aminmax(layernorm2_b))
    #     layernorm2_o = activations[f"{prefix}.layer.{l}.output.LayerNorm"]["output"][0]
    #     plot_schemes(layernorm2_o, f'layernorm2_o_{l}')
    #     # print(torch.aminmax(layernorm2_o))

    # exit()

    ## negative weights and activations - symmetric quantization scheme!
    # print(inputs)
    # observers = [ # qscheme=torch.per_tensor_symmetric
    #     # MinMaxObserver(qscheme=torch.per_tensor_symmetric), 
    #     MovingAverageMinMaxObserver(qscheme=torch.per_tensor_symmetric), 
    #     MovingAverageMinMaxObserver(qscheme=torch.per_tensor_affine)
    #     # HistogramObserver(),
    # ]
    # qschemes = [torch.per_tensor_affine, torch.per_tensor_symmetric]
    # for observer in observers:
    #     observer(inputs.input_ids)
    #     print(f"{observer.__class__.__name__}| ")

    # obs_sym = MovingAverageMinMaxObserver(qscheme=torch.per_tensor_symmetric) #, dtype=torch.qint8)
    # obs_aff = MovingAverageMinMaxObserver(qscheme=torch.per_tensor_affine) #, dtype=torch.qint8)
    # obs_sym(inputs.input_ids)
    # obs_aff(inputs.input_ids)
    # print(f'moving avg min/max, sym: {obs_sym.calculate_qparams()}')
    # print(f'moving avg min/max, aff: {obs_aff.calculate_qparams()}')

    # ## USING SYMMETRIC DUE TO NEGATIVE ACTIVATIONS!!

    # backend = "fbgemm"
    qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
        weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric)
    )

    ## EAGER MODE
    # import copy
    # model = copy.deepcopy(model_fp32)
    # model.eval()
    # """Fuse
    # - Inplace fusion replaces the first module in the sequence with the fused module, and the rest with identity modules
    # """
    # torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
    # torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

    model_int8 = BertForSequenceClassification(quantization=True, config=model_fp32.config).from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity",
    )
    # print(model_int8)
    # print(model_int8.bert.encoder.layer[0].attention.self.query.weight.flatten())
    # print(model_fp32.bert.encoder.layer[0].attention.self.query.weight.flatten())
    # exit()

    # model_int8.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_int8.qconfig = qconfig
    # print(model.qconfig)
    # exit()
    torch.quantization.prepare(model_int8, inplace=True)

    dataset = load_dataset('yelp_review_full')
    dataloader = DataLoader(
        dataset['test'],
        batch_size=1,
        shuffle=True,
        num_workers=1
    )
    # TOKENIZERS_PARALLELISM = False

    from tqdm import tqdm
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader)):
            if i == 500: break
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
            model_int8(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)  
        
    torch.quantization.convert(model_int8.bert, inplace=True)

    print(model_int8.bert.encoder.layer[0].attention.self.query.weight.element_size()) # 1 byte instead of 4 bytes for FP32

    # my_qconfig = torch.quantization.QConfig(
    #     activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
    #     weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric)
    # )

    # model_int8 = torch.ao.quantization.quantize_dynamic(
    #     model_fp32,
    #     {torch.nn.Linear},
    #     dtype=torch.qint8
    # )

    # sentence = "my cat is the cutest cat ever!"
    # inputs = tokenizer(sentence, return_tensors="pt").to(device)
    # outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, labels=torch.tensor([1]))
    # predict = outputs.logits.argmax().item()
    # loss = outputs.loss
    # print(f'{sentence} --> {num_to_label[predict]}')
    # print(f'loss: {loss}')

    # sentence = "i do not like running outside in the wind and cold."
    # inputs = tokenizer(sentence, return_tensors="pt").to(device)
    # outputs = model(**inputs, labels=torch.tensor([0]))
    # predict = outputs.logits.argmax().item()
    # loss = outputs.loss
    # print(f'{sentence} --> {num_to_label[predict]}')
    # print(f'loss: {loss}')


if __name__ == '__main__':
    main()