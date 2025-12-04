import math
import pdb
import time
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import argparse

from os import makedirs
from os.path import exists
from transformers import BertTokenizerFast
from modeling_bert_local import BertForSequenceClassification
from modeling_enc_dec_local import EncoderDecoderModel


class Identity(nn.Module):
    def forward(self, input):
        return input


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def disable_layer_norm(m):
    m.reset_parameters()
    m.eval()
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.zero_()


def remove_layernorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            disable_layer_norm(module)
        else:
            remove_layernorm(module)


def split(weight, num_accel):
    w_rows, w_cols = weight.shape
    if w_rows % num_accel != 0:
        print('matrix cannot be split evently amongst accelerators')
        exit()
    block = w_rows // num_accel
    blocks = torch.zeros((num_accel, block, w_cols))
    for i in range(num_accel):     
        blocks[i] = weight[i * block : (i + 1) * block, :]
    return blocks


def split_mat_mul_batched(input, weight, bias, num_accel):
    bs, i_rows, i_cols = input.shape
    w_rows, w_cols = weight.shape
    if w_rows % num_accel != 0:
        print('matrix cannot be split evently amongst accelerators')
        exit()
    block = w_rows // num_accel
    output = torch.zeros((bs, i_rows, w_rows))
    for i in range(num_accel):
        output[:, :, i * block : (i + 1) * block] = torch.einsum(
            "ijk,lk->ijl",
            input, 
            weight[i * block : (i + 1) * block, :]
        )
    if bias is not None:
        output += bias
    return output


def split_mat_mul_unbatched(input, weight, bias, num_accel):
    i_rows, i_cols = input.shape
    w_rows, w_cols = weight.shape
    if w_rows % num_accel != 0:
        print('matrix cannot be split evently amongst accelerators')
        exit()
    block = w_rows // num_accel
    output = torch.zeros((i_rows, w_rows))
    for i in range(num_accel):
        output[:, i * block : (i + 1) * block] = torch.einsum(
            "jk,lk->jl",
            input, 
            weight[i * block : (i + 1) * block, :]
        )
    if bias is not None:
        output += bias
    return output


def compare_tensors(ten1, ten2, diff):
    ## returns false if any value in the tensors differ by greater than diff
    ## returns true otherwise, to account for precision differences in parallel
    ## versus normal matmul computations (floating point round-off error...)
    ten1 = ten1.flatten()
    ten2 = ten2.flatten()
    diffs = (ten1 != ten2).nonzero().flatten()
    flag = True
    for index in diffs:
        if abs(ten1[index] - ten2[index]) > diff:
            # print('%.3f'%(ten1[index].item()), '%.3f'%(ten2[index].item()))
            flag = False
            return flag
    return flag


def scratch_sdpa_masked_unbatched(query, key, value, mask):
    seq_length, hidden_dim = query.shape
    num_attention_heads = 12
    attention_head_size = hidden_dim // num_attention_heads

    new_q_shape = query.size()[:-1] + (num_attention_heads, attention_head_size)
    query = query.view(new_q_shape).permute(1, 0, 2)
    new_k_shape = key.size()[:-1] + (num_attention_heads, attention_head_size)
    key = key.view(new_k_shape).permute(1, 0, 2)
    new_v_shape = value.size()[:-1] + (num_attention_heads, attention_head_size)
    value = value.view(new_v_shape).permute(1, 0, 2)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores /= math.sqrt(attention_head_size)
    target_len = attention_scores.shape[2]

    extend_mask = mask[:, None, :].expand(1, seq_length, target_len).repeat(num_attention_heads, 1, 1)
    boolean_mask = (extend_mask == 0)
    extend_mask = extend_mask.masked_fill_(boolean_mask, -1e9)
    attention_scores += extend_mask

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_probs, value)
    attention_output = attention_output.permute(1, 0, 2).reshape(seq_length, hidden_dim)
    return (attention_output, attention_scores)


def scratch_sdpa_masked_batched(query, key, value, mask):
    batch_size, seq_length, hidden_dim = query.shape
    num_attention_heads = 12
    attention_head_size = hidden_dim // num_attention_heads

    new_q_shape = query.size()[:-1] + (num_attention_heads, attention_head_size)
    query = query.view(new_q_shape).permute(0, 2, 1, 3)
    new_k_shape = key.size()[:-1] + (num_attention_heads, attention_head_size)
    key = key.view(new_k_shape).permute(0, 2, 1, 3)
    new_v_shape = value.size()[:-1] + (num_attention_heads, attention_head_size)
    value = value.view(new_v_shape).permute(0, 2, 1, 3)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores /= math.sqrt(attention_head_size)
    target_len = attention_scores.shape[3]

    extend_mask = mask[:, None, None, :].expand(batch_size, 1, seq_length, target_len).repeat(1, num_attention_heads, 1, 1)
    boolean_mask = (extend_mask == 0)
    extend_mask = extend_mask.masked_fill_(boolean_mask, -1e9)
    attention_scores += extend_mask

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_probs, value)
    attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, hidden_dim)
    return (attention_output, attention_scores)


def dump_bert_parallel(model, inputs, save, verbose, dropout, home, num_accel):
    model = model.to("cpu")
    inputs = inputs.to("cpu")

    if save:
        if not exists(home):
            makedirs(home)

    # def disable_dropout_prompt():
    #     response = input('do you want to disable dropout layers? (y/n) ')
    #     if response == 'y':
    #         disable_dropout(model)
    #         print('disabling dropout layers')
    #     elif response == 'n':
    #         print('leaving dropout layers enabled')
    #     else:
    #         print('please respond with y or n - other inputs will not be parsed correctly')
    #         disable_dropout_prompt()
    # disable_dropout_prompt()

    # def num_accel_prompt():
    #     num_accel = 1
    #     response = input('how many accelerators do you want to employ output parallelism for? (1-10) ') ## check this number with jenna
    #     if int(response) in range(1, 11):
    #         num_accel = int(response)
    #         print(f'setting number of accelerators to {num_accel}')
    #         return num_accel
    #     else:
    #         print('please respond with a number 1 through 10 - other inputs will not be parsed correctly')
    #         num_accel_prompt()
    # num_accel = num_accel_prompt()
    
    # def verbose_prompt():
    #     verbose = False
    #     response = input('do you want verbose output? (y/n) ')
    #     if response == 'y':
    #         verbose = True
    #         print('enabling verbose output - layer sizes will be printed')
    #         return verbose
    #     elif response == 'n':
    #         print('continuing without verbose output - layer sizes will not be printed')
    #         return verbose
    #     else:
    #         print('please respond with y or n - other inputs will not be parsed correctly')
    #         verbose_prompt()
    # verbose = verbose_prompt()

    # def save_prompt():
    #     save = False
    #     response = input('do you want to save all mat-mul weights? (y/n) ')
    #     if response == 'y':
    #         save = True
    #         print(f'weights/activations will be saved to "{home}/model/bert_base/npy"')
    #         print('home location can be modified within the script')
    #         return save
    #     elif response == 'n':
    #         print('weights/activations will not be saved')
    #         return save
    #     else:
    #         print('please respond with y or n - other inputs will not be parsed correctly')
    #         save_prompt()
    # save = save_prompt()

    if not dropout:
        disable_dropout(model)

    def register_hooks(model):
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = {'input': input, 'output': output}
            return hook
        for name, module in model.named_modules():
            module.register_forward_hook(get_activation(name))
        return activations
    activations = register_hooks(model)

    with torch.no_grad():
        # model(**inputs)
        model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)

    ## iterate over encoder
    for l in range(12):
        block = model.encoder.encoder.layer[l]

        '''
        (attention): BertAttention(
            (self): BertSdpaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0, inplace=False)
                (sdpa_masked): ScratchSDPAMasked()
            )
            (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0, inplace=False)
            )
        )
        (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0, inplace=False)
        )
        '''

        ## input to self-attention block of encoder
        attn_in = activations[f"encoder.encoder.layer.{l}.attention.self"]["input"][0]
        
        ## query weights, bias, model outputs, local outputs
        query_w = block.attention.self.query.weight
        query_b = block.attention.self.query.bias
        query_o = activations[f"encoder.encoder.layer.{l}.attention.self.query"]["output"][0]
        # if len(query_o.shape) < 3:
        #     ## batch it
        #     query_o = query_o.unsqueeze(0)
        sw_query_o = torch.einsum("ijk,lk->ijl", attn_in, query_w) + query_b
        split_query_w = split(query_w, num_accel)

        ## key weights, bias, model outputs, local outputs
        key_w = block.attention.self.key.weight
        key_b = block.attention.self.key.bias
        key_o = activations[f"encoder.encoder.layer.{l}.attention.self.key"]["output"][0]
        # if len(key_o.shape) < 3:
        #     ## batch it
        #     key_o = key_o.unsqueeze(0)
        sw_key_o = torch.einsum("ijk,lk->ijl", attn_in, key_w) + key_b
        split_key_w = split(key_w, num_accel)

        ## value weights, bias, model outputs, local outputs
        value_w = block.attention.self.value.weight
        value_b = block.attention.self.value.bias
        value_o = activations[f"encoder.encoder.layer.{l}.attention.self.value"]["output"][0]
        # if len(value_o.shape) < 3:
        #     ## batch it
        #     value_o = value_o.unsqueeze(0)
        sw_value_o = torch.einsum("ijk,lk->ijl", attn_in, value_w) + value_b
        split_value_w = split(value_w, num_accel)
        
        ## self-attention outputs and local scratch SDPA attention results
        attn_o = activations[f"encoder.encoder.layer.{l}.attention.self"]["output"][0]
        sw_attn_o, sw_scores_o = scratch_sdpa_masked_batched(sw_query_o, sw_key_o, sw_value_o, inputs["attention_mask"])

        ## first output network
        # dense_in = activations[f"encoder.encoder.layer.{l}.attention.output.dense"]["input"][0]
        dense_w = block.attention.output.dense.weight
        dense_b = block.attention.output.dense.bias
        dense_o = activations[f"encoder.encoder.layer.{l}.attention.output.dense"]["output"][0]
        # if len(dense_o.shape) < 3:
        #     ## batch it
        #     dense_o = dense_o.unsqueeze(0)
        sw_dense_o = torch.einsum("ijk,lk->ijl", sw_attn_o, dense_w) + dense_b
        split_dense_w = split(dense_w, num_accel)
        
        ## first residual and layernorm
        rsd1_o = dense_o + attn_in
        sw_rsd1_o = sw_dense_o + attn_in
        layernorm1_o = activations[f"encoder.encoder.layer.{l}.attention.output.LayerNorm"]["output"][0]
        # if len(layernorm1_o.shape) < 3:
        #     ## batch it
        #     layernorm1_o = layernorm1_o.unsqueeze(0)
        layernorm1_w = block.attention.output.LayerNorm.weight
        layernorm1_b = block.attention.output.LayerNorm.bias

        ## intermediate network
        ffn1_w = block.intermediate.dense.weight
        ffn1_b = block.intermediate.dense.bias
        gelu_in = activations[f"encoder.encoder.layer.{l}.intermediate.intermediate_act_fn"]["input"][0]
        sw_gelu_in = torch.einsum("jk,lk->jl", layernorm1_o, ffn1_w) + ffn1_b
        ffn1_o_gelu = activations[f"encoder.encoder.layer.{l}.intermediate.intermediate_act_fn"]["output"][0]
        # if len(ffn1_o_gelu.shape) < 3:
        #     ## batch it
        #     ffn1_o_gelu = ffn1_o_gelu.unsqueeze(0)
        sw_ffn1_o_gelu = torch.nn.functional.gelu(torch.einsum("jk,lk->jl", layernorm1_o, ffn1_w) + ffn1_b)
        split_ffn1_w = split(ffn1_w, num_accel)

        ## second output network
        ffn2_w = block.output.dense.weight
        ffn2_b = block.output.dense.bias
        ffn2_o = activations[f"encoder.encoder.layer.{l}.output.dense"]["output"][0]
        # if len(ffn2_o.shape) < 3:
        #     ## batch it
        #     ffn2_o = ffn2_o.unsqueeze(0)
        sw_ffn2_o = torch.einsum("jk,lk->jl", sw_ffn1_o_gelu, ffn2_w) + ffn2_b
        split_ffn2_w = split(ffn2_w, num_accel)

        ## second residual and layernorm
        rsd2_o = ffn2_o + layernorm1_o    
        sw_rsd2_o = sw_ffn2_o + layernorm1_o     
        layernorm2_o = activations[f"encoder.encoder.layer.{l}.output.LayerNorm"]["output"][0]
        # if len(layernorm2_o.shape) < 3:
        #     ## batch it
        #     layernorm2_o = layernorm2_o.unsqueeze(0)
        layernorm2_w = block.output.LayerNorm.weight
        layernorm2_b = block.output.LayerNorm.bias

        ## verify output parallelism
        split_query_o = split_mat_mul_batched(attn_in, query_w, query_b, num_accel)
        split_key_o = split_mat_mul_batched(attn_in, key_w, key_b, num_accel)
        split_value_o = split_mat_mul_batched(attn_in, value_w, value_b, num_accel)
        split_dense_o = split_mat_mul_batched(sw_attn_o, dense_w, dense_b, num_accel)
        split_ffn1_o_gelu = torch.nn.functional.gelu(split_mat_mul_unbatched(layernorm1_o, ffn1_w, ffn1_b, num_accel))
        split_ffn2_o = split_mat_mul_unbatched(sw_ffn1_o_gelu, ffn2_w, ffn2_b, num_accel)
        query_verif = compare_tensors(sw_query_o, split_query_o, 1e-4)
        key_verif = compare_tensors(sw_key_o, split_key_o, 1e-4)
        value_verif = compare_tensors(sw_value_o, split_value_o, 1e-4)
        dense_verif = compare_tensors(sw_dense_o, split_dense_o, 1e-4)
        ffn1_verif = compare_tensors(sw_ffn1_o_gelu, split_ffn1_o_gelu, 1e-4)
        ffn2_verif = compare_tensors(sw_ffn2_o, split_ffn2_o, 1e-4)

        if verbose:
            print(f'---- START ENCODER BLOCK {l} ----')
            print(f'----> query')
            print(f'query_o \t\t {list(query_o.shape)}')
            print(f'sw_query_o \t\t {list(sw_query_o.shape)}')
            print(f'output parallelism match: {query_verif}')
            print(f'----> key')
            print(f'key_o \t\t\t {list(key_o.shape)}')
            print(f'sw_key_o \t\t {list(sw_key_o.shape)}')
            print(f'output parallelism match: {key_verif}')
            print(f'----> value')
            print(f'value_o \t\t {list(value_o.shape)}')
            print(f'sw_value_o \t\t {list(sw_value_o.shape)}')
            print(f'output parallelism match: {value_verif}')
            print(f'----> attention')
            print(f'attn_o \t\t\t {list(attn_o.shape)}')
            print(f'sw_attn_o \t\t {list(sw_attn_o.shape)}')
            print(f'----> dense')
            print(f'dense_o \t\t {list(dense_o.shape)}')
            print(f'sw_dense_o \t\t {list(sw_dense_o.shape)}')
            print(f'output parallelism match: {dense_verif}')
            print(f'----> residual + layernorm 1')
            print(f'rsd1_o \t\t\t {list(rsd1_o.shape)}')
            print(f'sw_rsd1_o \t\t {list(sw_rsd1_o.shape)}')
            print(f'layernorm1_o \t {list(layernorm1_o.shape)}')
            print(f'----> ffn1')
            print(f'ffn1_o_gelu \t {list(ffn1_o_gelu.shape)}')
            print(f'sw_ffn1_o_gelu \t {list(sw_ffn1_o_gelu.shape)}')
            print(f'output parallelism match: {ffn1_verif}')
            print(f'gelu_in \t\t {list(gelu_in.shape)}')
            print(f'sw_gelu_in \t\t {list(sw_gelu_in.shape)}')
            print(f'----> ffn2')
            print(f'ffn2_o \t\t\t {list(ffn2_o.shape)}')
            print(f'sw_ffn2_o \t\t {list(sw_ffn2_o.shape)}')
            print(f'output parallelism match: {ffn2_verif}')
            print(f'----> residual + layernorm 2')
            print(f'rsd2_o \t\t\t {list(rsd2_o.shape)}')
            print(f'sw_rsd2_o \t\t {list(sw_rsd2_o.shape)}')
            print(f'layernorm2_o \t {list(layernorm2_o.shape)}')
            print(f'----  END ENCODER BLOCK {l}  ----\n')

        if save:
            np.save(f"{home}/encoder_l{l}_attn_in", attn_in.detach().numpy())
            np.save(f"{home}/encoder_l{l}_query_w", query_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_query_w_t", query_w.T.detach().numpy().flatten())
            np.save(f"{home}/encoder_l{l}_split_{num_accel}_query_w", split_query_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_query_b", query_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_query_o", query_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_key_w", key_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_key_w_t", key_w.T.detach().numpy().flatten())
            np.save(f"{home}/encoder_l{l}_split_{num_accel}_key_w", split_key_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_key_b", key_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_key_o", key_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_value_w", value_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_value_w_t", value_w.T.detach().numpy().flatten())
            np.save(f"{home}/encoder_l{l}_split_{num_accel}_value_w", split_value_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_value_b", value_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_value_o", value_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_attn_o", attn_o.detach().numpy())    
            np.save(f"{home}/encoder_l{l}_sw_scores_o", sw_scores_o.detach().numpy())    
            np.save(f"{home}/encoder_l{l}_dense_w", dense_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_dense_w_t", dense_w.T.detach().numpy().flatten())
            np.save(f"{home}/encoder_l{l}_split_{num_accel}_dense_w", split_dense_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_dense_b", dense_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_dense_o", dense_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_sw_rsd1_o", rsd1_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_lyn1_o", layernorm1_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_lyn1_w", layernorm1_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_lyn1_b", layernorm1_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn1_w", ffn1_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn1_w_t", ffn1_w.T.detach().numpy().flatten())
            np.save(f"{home}/encoder_l{l}_split_{num_accel}_ffn1_w", split_ffn1_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn1_b", ffn1_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn1_o", ffn1_o_gelu.detach().numpy())
            np.save(f"{home}/encoder_l{l}_gelu_in", gelu_in.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn2_w", ffn2_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn2_w_t", ffn2_w.T.detach().numpy().flatten())
            np.save(f"{home}/encoder_l{l}_split_{num_accel}_ffn2_w", split_ffn2_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn2_b", ffn2_b.detach().numpy())
            np.save(f"{home}/encoder_l{l}_ffn2_o", ffn2_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_sw_rsd2_o", rsd2_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_lyn2_o", layernorm2_o.detach().numpy())
            np.save(f"{home}/encoder_l{l}_lyn2_w", layernorm2_w.detach().numpy())
            np.save(f"{home}/encoder_l{l}_lyn2_b", layernorm2_b.detach().numpy())
    
    ## iterate over decoder
    for l in range(12):
        block = model.decoder.bert.encoder.layer[l]

        '''
        (attention): BertAttention(
            (self): BertSdpaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0, inplace=False)
                (sdpa_masked): ScratchSDPAMasked()
            )
            (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0, inplace=False)
            )
        )
        (crossattention): BertAttention(
            (self): BertSdpaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0, inplace=False)
                (sdpa_masked): ScratchSDPAMasked()
            )
            (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0, inplace=False)
            )
        )
        (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0, inplace=False)
        )
        '''

        ## input to self-attention block of encoder
        slf_attn_in = activations[f"decoder.bert.encoder.layer.{l}.attention.self"]["input"][0]
        slf_sdpa_attn_in = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self.sdpa_masked"]["input"]
        slf_query_o = slf_sdpa_attn_in[0]
        slf_key_o = slf_sdpa_attn_in[1]
        slf_value_o = slf_sdpa_attn_in[2]
        
        ## query weights, bias, model outputs, local outputs
        slf_query_w = block.attention.self.query.weight
        slf_query_b = block.attention.self.query.bias
        # slf_query_o = activations[f"decoder.bert.encoder.layer.{l}.attention.self.query"]["output"][0]
        # slf_sw_query_o = torch.einsum("ijk,lk->ijl", slf_attn_in, slf_query_w) + slf_query_b
        slf_split_query_w = split(slf_query_w, num_accel)

        ## key weights, bias, model outputs, local outputs
        slf_key_w = block.attention.self.key.weight
        slf_key_b = block.attention.self.key.bias
        # slf_key_o = activations[f"decoder.bert.encoder.layer.{l}.attention.self.key"]["output"][0]
        # slf_sw_key_o = torch.einsum("ijk,lk->ijl", slf_attn_in, slf_key_w) + slf_key_b
        slf_split_key_w = split(slf_key_w, num_accel)

        ## value weights, bias, model outputs, local outputs
        slf_value_w = block.attention.self.value.weight
        slf_value_b = block.attention.self.value.bias
        # slf_value_o = activations[f"decoder.bert.encoder.layer.{l}.attention.self.value"]["output"][0]
        # slf_sw_value_o = torch.einsum("ijk,lk->ijl", slf_attn_in, slf_value_w) + slf_value_b
        slf_split_value_w = split(slf_value_w, num_accel)
        
        ## self-attention outputs and local scratch SDPA attention results
        slf_attn_o = activations[f"decoder.bert.encoder.layer.{l}.attention.self"]["output"][0]
        slf_attn_pkv_o = activations[f"decoder.bert.encoder.layer.{l}.attention.self"]["output"][1]
        # slf_sw_attn_o = scratch_sdpa_masked(slf_sw_query_o, slf_sw_key_o, slf_sw_value_o, inputs["attention_mask"])
        slf_sw_attn_o, slf_sw_scores_o = scratch_sdpa_masked_batched(slf_query_o, slf_key_o, slf_value_o, inputs["attention_mask"])

        ## self-attention output network
        slf_dense_in = activations[f"decoder.bert.encoder.layer.{l}.attention.output.dense"]["input"][0]
        slf_dense_w = block.attention.output.dense.weight
        slf_dense_b = block.attention.output.dense.bias
        slf_dense_o = activations[f"decoder.bert.encoder.layer.{l}.attention.output.dense"]["output"][0]
        slf_sw_dense_o = torch.einsum("ijk,lk->ijl", slf_sw_attn_o, slf_dense_w) + slf_dense_b
        slf_split_dense_w = split(slf_dense_w, num_accel)
        
        ## self-attention residual and layernorm
        slf_rsd1_o = slf_dense_o + slf_attn_in
        slf_sw_rsd1_o = slf_sw_dense_o + slf_attn_in
        slf_layernorm1_o = activations[f"decoder.bert.encoder.layer.{l}.attention.output.LayerNorm"]["output"][0]
        slf_layernorm1_w = block.attention.output.LayerNorm.weight
        slf_layernorm1_b = block.attention.output.LayerNorm.bias

        ## verify output parallelism
        # slf_split_query_o = split_mat_mul_batched(slf_attn_in, slf_query_w, slf_query_b, num_accel)
        # slf_split_key_o = split_mat_mul_batched(slf_attn_in, slf_key_w, slf_key_b, num_accel)
        # slf_split_value_o = split_mat_mul_batched(slf_attn_in, slf_value_w, slf_value_b, num_accel)
        slf_split_dense_o = split_mat_mul_batched(slf_sw_attn_o, slf_dense_w, slf_dense_b, num_accel)
        # slf_query_verif = compare_tensors(slf_sw_query_o, slf_split_query_o, 1e-4)
        # slf_key_verif = compare_tensors(slf_sw_key_o, slf_split_key_o, 1e-4)
        # slf_value_verif = compare_tensors(slf_sw_value_o, slf_split_value_o, 1e-4)
        slf_dense_verif = compare_tensors(slf_sw_dense_o, slf_split_dense_o, 1e-4)

        ## input to cross-attention block of encoder
        crs_attn_in = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self"]["input"][0]
        crs_sdpa_attn_in = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self.sdpa_masked"]["input"]
        crs_query_o = crs_sdpa_attn_in[0]
        crs_key_o = crs_sdpa_attn_in[1]
        crs_value_o = crs_sdpa_attn_in[2]
        
        ## query weights, bias, model outputs, local outputs
        crs_query_w = block.crossattention.self.query.weight
        crs_query_b = block.crossattention.self.query.bias
        # crs_query_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self.query"]["output"][0]
        # crs_sw_query_o = torch.einsum("ijk,lk->ijl", crs_attn_in, crs_query_w) + query_b
        crs_split_query_w = split(crs_query_w, num_accel)

        ## key weights, bias, model outputs, local outputs
        crs_key_w = block.crossattention.self.key.weight
        crs_key_b = block.crossattention.self.key.bias
        # crs_key_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self.key"]["output"][0]
        # crs_sw_key_o = torch.einsum("ijk,lk->ijl", attn_in, key_w) + key_b
        crs_split_key_w = split(crs_key_w, num_accel)

        ## value weights, bias, model outputs, local outputs
        crs_value_w = block.crossattention.self.value.weight
        crs_value_b = block.crossattention.self.value.bias
        # crs_value_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self.value"]["output"][0]
        # crs_sw_value_o = torch.einsum("ijk,lk->ijl", attn_in, value_w) + value_b
        crs_split_value_w = split(crs_value_w, num_accel)
        
        ## cross-attention outputs and local scratch SDPA attention results
        crs_attn_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self"]["output"][0]
        crs_attn_pkv_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.self"]["output"][1]
        crs_sw_attn_o, crs_sw_scores_o = scratch_sdpa_masked_batched(crs_query_o, crs_key_o, crs_value_o, inputs["attention_mask"])

        ## cross-attention output network
        crs_dense_in = activations[f"decoder.bert.encoder.layer.{l}.crossattention.output.dense"]["input"][0]
        crs_dense_w = block.crossattention.output.dense.weight
        crs_dense_b = block.crossattention.output.dense.bias
        crs_dense_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.output.dense"]["output"][0]
        crs_sw_dense_o = torch.einsum("ijk,lk->ijl", crs_sw_attn_o, crs_dense_w) + crs_dense_b
        crs_split_dense_w = split(crs_dense_w, num_accel)
        
        ## cross-attention residual and layernorm
        crs_rsd1_o = crs_dense_o + crs_attn_in
        crs_sw_rsd1_o = crs_sw_dense_o + crs_attn_in
        crs_layernorm1_o = activations[f"decoder.bert.encoder.layer.{l}.crossattention.output.LayerNorm"]["output"][0]
        crs_layernorm1_w = block.crossattention.output.LayerNorm.weight
        crs_layernorm1_b = block.crossattention.output.LayerNorm.bias

        ## verify output parallelism
        # crs_split_query_o = split_mat_mul_batched(crs_attn_in, crs_query_w, crs_query_b, num_accel)
        # crs_split_key_o = split_mat_mul_batched(crs_attn_in, crs_key_w, crs_key_b, num_accel)
        # crs_split_value_o = split_mat_mul_batched(crs_attn_in, crs_value_w, crs_value_b, num_accel)
        crs_split_dense_o = split_mat_mul_batched(crs_sw_attn_o, crs_dense_w, crs_dense_b, num_accel)
        # crs_query_verif = compare_tensors(crs_sw_query_o, crs_split_query_o, 1e-4)
        # crs_key_verif = compare_tensors(crs_sw_key_o, crs_split_key_o, 1e-4)
        # crs_value_verif = compare_tensors(crs_sw_value_o, crs_split_value_o, 1e-4)
        crs_dense_verif = compare_tensors(crs_sw_dense_o, crs_split_dense_o, 1e-4)

        ## intermediate network
        ffn1_w = block.intermediate.dense.weight
        ffn1_b = block.intermediate.dense.bias
        gelu_in = activations[f"decoder.bert.encoder.layer.{l}.intermediate.intermediate_act_fn"]["input"][0]
        sw_gelu_in = torch.einsum("jk,lk->jl", crs_layernorm1_o, ffn1_w) + ffn1_b
        ffn1_o_gelu = activations[f"decoder.bert.encoder.layer.{l}.intermediate.intermediate_act_fn"]["output"][0].squeeze(0)
        sw_ffn1_o_gelu = torch.nn.functional.gelu(torch.einsum("jk,lk->jl", crs_layernorm1_o, ffn1_w) + ffn1_b)
        split_ffn1_w = split(ffn1_w, num_accel)

        ## second output network
        ffn2_w = block.output.dense.weight
        ffn2_b = block.output.dense.bias
        ffn2_o = activations[f"decoder.bert.encoder.layer.{l}.output.dense"]["output"][0]
        sw_ffn2_o = torch.einsum("jk,lk->jl", sw_ffn1_o_gelu, ffn2_w) + ffn2_b
        split_ffn2_w = split(ffn2_w, num_accel)

        ## second residual and layernorm
        rsd2_o = ffn2_o + crs_layernorm1_o
        sw_rsd2_o = sw_ffn2_o + crs_layernorm1_o     
        layernorm2_o = activations[f"decoder.bert.encoder.layer.{l}.output.LayerNorm"]["output"][0]
        layernorm2_w = block.output.LayerNorm.weight
        layernorm2_b = block.output.LayerNorm.bias

        ## verify output parallelism
        split_ffn1_o_gelu = torch.nn.functional.gelu(split_mat_mul_unbatched(crs_layernorm1_o, ffn1_w, ffn1_b, num_accel))
        split_ffn2_o = split_mat_mul_unbatched(sw_ffn1_o_gelu, ffn2_w, ffn2_b, num_accel)
        ffn1_verif = compare_tensors(sw_ffn1_o_gelu, split_ffn1_o_gelu, 1e-4)
        ffn2_verif = compare_tensors(sw_ffn2_o, split_ffn2_o, 1e-4)

        if verbose:
            print(f'---- START DECODER BLOCK {l} ----')
            print(f'----> slf_query')
            print(f'slf_query_o \t {list(slf_query_o.shape)}')
            # print(f'slf_sw_query_o  {list(slf_sw_query_o.shape)}')
            # print(f'slf_split_query_o  {list(slf_split_query_o.shape)}')
            print(f'----> slf_key')
            print(f'slf_key_o \t\t {list(slf_key_o.shape)}')
            # print(f'slf_sw_key_o  {list(slf_sw_key_o.shape)}')
            # print(f'slf_split_key_o  {list(slf_split_key_o.shape)}')
            print(f'----> slf_value')
            print(f'slf_value_o \t {list(slf_value_o.shape)}')
            # print(f'slf_sw_value_o  {list(slf_sw_value_o.shape)}')
            # print(f'slf_split_value_o  {list(slf_split_value_o.shape)}')
            print(f'----> slf_attention')
            print(f'slf_attn_o \t\t {list(slf_attn_o.shape)}')
            print(f'slf_sw_attn_o \t {list(slf_sw_attn_o.shape)}')
            print(f'slf_attn_pkv_o \t {list(slf_attn_pkv_o[0].shape)}')
            print(f'----> slf_dense')
            print(f'slf_dense_o \t {list(slf_dense_o.shape)}')
            print(f'slf_sw_dense_o \t {list(slf_sw_dense_o.shape)}')
            # print(f'slf_split_dense_o  {list(slf_split_dense_o.shape)}')
            print(f'output parallelism match: {slf_dense_verif}')
            print(f'----> slf_residual + slf_layernorm 1')
            print(f'slf_rsd1_o \t\t {list(slf_rsd1_o.shape)}')
            print(f'slf_sw_rsd1_o \t {list(slf_sw_rsd1_o.shape)}')
            print(f'slf_layernorm1_o {list(slf_layernorm1_o.shape)}')
            print(f'----> crs_query')
            print(f'crs_query_o \t {list(crs_query_o.shape)}')
            # print(f'crs_sw_query_o  {list(crs_sw_query_o.shape)}')
            # print(f'crs_split_query_o  {list(crs_split_query_o.shape)}')
            print(f'----> crs_key')
            print(f'crs_key_o \t\t {list(crs_key_o.shape)}')
            # print(f'crs_sw_key_o  {list(crs_sw_key_o.shape)}')
            # print(f'crs_split_key_o  {list(crs_split_key_o.shape)}')
            print(f'----> crs_value')
            print(f'crs_value_o \t {list(crs_value_o.shape)}')
            # print(f'crs_sw_value_o  {list(crs_sw_value_o.shape)}')
            # print(f'crs_split_value_o  {list(crs_split_value_o.shape)}')
            print(f'----> crs_attention')
            print(f'crs_attn_o \t\t {list(crs_attn_o.shape)}')
            print(f'crs_sw_attn_o \t {list(crs_sw_attn_o.shape)}')
            print(f'crs_attn_pkv_o \t {list(crs_attn_pkv_o[0].shape)}')
            print(f'----> crs_dense')
            print(f'crs_dense_o \t {list(crs_dense_o.shape)}')
            print(f'crs_sw_dense_o \t {list(crs_sw_dense_o.shape)}')
            # print(f'crs_split_dense_o  {list(crs_split_dense_o.shape)}')
            print(f'output parallelism match: {crs_dense_verif}')
            print(f'----> crs_residual + crs_layernorm 1')
            print(f'crs_rsd1_o \t\t {list(crs_rsd1_o.shape)}')
            print(f'crs_sw_rsd1_o \t {list(crs_sw_rsd1_o.shape)}')
            print(f'crs_layernorm1_o {list(crs_layernorm1_o.shape)}')
            print(f'----> ffn1')
            print(f'ffn1_o_gelu \t {list(ffn1_o_gelu.shape)}')
            print(f'sw_ffn1_o_gelu \t {list(sw_ffn1_o_gelu.shape)}')
            # print(f'split_ffn1_o_gelu {list(split_ffn1_o_gelu.shape)}')
            print(f'output parallelism match: {ffn1_verif}')
            print(f'gelu_in \t\t {list(gelu_in.shape)}')
            print(f'sw_gelu_in \t\t {list(sw_gelu_in.shape)}')
            print(f'----> ffn2')
            print(f'ffn2_o \t\t\t {list(ffn2_o.shape)}')
            print(f'sw_ffn2_o \t\t {list(sw_ffn2_o.shape)}')
            # print(f'split_ffn2_o  {list(split_ffn2_o.shape)}')
            print(f'output parallelism match: {ffn2_verif}')
            print(f'----> residual + layernorm 2')
            print(f'sw_rsd2_o \t\t {list(sw_rsd2_o.shape)}')
            print(f'layernorm2_o \t {list(layernorm2_o.shape)}')
            print(f'----  END DECODER BLOCK {l}  ----\n')

        if save:
            np.save(f"{home}/decoder_l{l}_slf_attn_in", slf_attn_in.detach().numpy())
            # np.save(f"{home}/decoder_l{l}_slf_sdpa_attn_in", slf_sdpa_attn_in.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_query_w", slf_query_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_query_w_t", slf_query_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_slf_split_{num_accel}_query_w", slf_split_query_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_query_b", slf_query_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_query_o", slf_query_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_key_w", slf_key_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_key_w_t", slf_key_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_slf_split_{num_accel}_key_w", slf_split_key_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_key_b", slf_key_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_key_o", slf_key_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_value_w", slf_value_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_value_w_t", slf_value_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_slf_split_{num_accel}_value_w", slf_split_value_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_value_b", slf_value_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_value_o", slf_value_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_attn_o", slf_attn_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_sw_scores_o", slf_sw_scores_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_dense_w", slf_dense_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_dense_w_t", slf_dense_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_slf_split_{num_accel}_dense_w", slf_split_dense_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_dense_b", slf_dense_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_dense_o", slf_dense_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_rsd1_o", slf_rsd1_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_lyn1_o", slf_layernorm1_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_lyn1_w", slf_layernorm1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_slf_lyn1_b", slf_layernorm1_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_attn_in", crs_attn_in.detach().numpy())
            # np.save(f"{home}/decoder_l{l}_crs_sdpa_attn_in", crs_sdpa_attn_in.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_query_w", crs_query_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_query_w_t", crs_query_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_crs_split_{num_accel}_query_w", crs_split_query_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_query_b", crs_query_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_query_o", crs_query_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_key_w", crs_key_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_key_w_t", crs_key_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_crs_split_{num_accel}_key_w", crs_split_key_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_key_b", crs_key_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_key_o", crs_key_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_value_w", crs_value_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_value_w_t", crs_value_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_crs_split_{num_accel}_value_w", crs_split_value_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_value_b", crs_value_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_value_o", crs_value_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_attn_o", crs_attn_o.detach().numpy())        
            np.save(f"{home}/decoder_l{l}_crs_sw_scores_o", crs_sw_scores_o.detach().numpy())       
            np.save(f"{home}/decoder_l{l}_crs_dense_w", crs_dense_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_dense_w_t", crs_dense_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_crs_split_{num_accel}_dense_w", crs_split_dense_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_dense_b", crs_dense_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_dense_o", crs_dense_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_rsd1_o", crs_rsd1_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_lyn1_o", crs_layernorm1_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs_lyn1_w", crs_layernorm1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_crs__lyn1_b", crs_layernorm1_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_w", ffn1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_w_t", ffn1_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_ffn1_w", split_ffn1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_b", ffn1_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_o", ffn1_o_gelu.detach().numpy())
            np.save(f"{home}/decoder_l{l}_gelu_in", gelu_in.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_w", ffn2_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_w_t", ffn2_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_ffn2_w", split_ffn2_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_b", ffn2_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_o", ffn2_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_rsd2_o", rsd2_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn2_o", layernorm2_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn2_w", layernorm2_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn2_b", layernorm2_b.detach().numpy())


if __name__ == "__main__":
    # model = BertForSequenceClassification.from_pretrained(
    #     "textattack/bert-base-uncased-yelp-polarity", 
    # )
    # tokenizer = BertTokenizerFast.from_pretrained(
    #     "textattack/bert-base-uncased-yelp-polarity", 
    # )
    # inputs = tokenizer(
    #     "my cat is the cutest cat ever!", 
    #     return_tensors="pt"
    # )
    model = EncoderDecoderModel.from_pretrained(
        "patrickvonplaten/bert2bert_cnn_daily_mail"
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        "patrickvonplaten/bert2bert_cnn_daily_mail"
    )
    article = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    inputs = tokenizer(
        article, 
        return_tensors="pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dropout', help='enable dropout layers in model', action='store_true')
    parser.add_argument('-s', '--save', help='save weights, activations, and attention scores', action='store_true')
    parser.add_argument('-v', '--verbose', help='print layer sizes and block numbers', action='store_true')
    parser.add_argument('-a', '--accel', type=int, help='number of accelerators to employ output parallelism', default=1)
    parser.add_argument('-l', '--location', type=str, help='location of home directory for saving', required=True)
    args = parser.parse_args()
    dump_bert_parallel(
        model=model, 
        inputs=inputs, 
        save=args.save, 
        verbose=args.verbose, 
        dropout=args.dropout, 
        home=args.location, 
        num_accel=args.accel
    )
    ## run me!
    ## python3 bert_base_output_parallel.py -l /example/folder/to/save -a 6 -v -s