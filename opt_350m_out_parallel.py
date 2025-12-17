import math
import pdb
import time
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import argparse

from os import makedirs, getcwd
from os.path import exists
from transformers import AutoTokenizer

# import sys
# sys.path.insert(1, f'~/dev/repos/ccqw/models')

from models import OPTForCausalLM


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


# def scratch_sdpa_masked_unbatched(query, key, value, mask):
#     seq_length, hidden_dim = query.shape
#     num_attention_heads = 12
#     attention_head_size = hidden_dim // num_attention_heads

#     new_q_shape = query.size()[:-1] + (num_attention_heads, attention_head_size)
#     query = query.view(new_q_shape).permute(1, 0, 2)
#     new_k_shape = key.size()[:-1] + (num_attention_heads, attention_head_size)
#     key = key.view(new_k_shape).permute(1, 0, 2)
#     new_v_shape = value.size()[:-1] + (num_attention_heads, attention_head_size)
#     value = value.view(new_v_shape).permute(1, 0, 2)

#     attention_scores = torch.matmul(query, key.transpose(-1, -2))
#     attention_scores /= math.sqrt(attention_head_size)
#     target_len = attention_scores.shape[2]

#     extend_mask = mask[:, None, :].expand(1, seq_length, target_len).repeat(num_attention_heads, 1, 1)
#     boolean_mask = (extend_mask == 0)
#     extend_mask = extend_mask.masked_fill_(boolean_mask, -1e9)
#     attention_scores += extend_mask

#     attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#     attention_output = torch.matmul(attention_probs, value)
#     attention_output = attention_output.permute(1, 0, 2).reshape(seq_length, hidden_dim)
#     return (attention_output, attention_scores)


def scratch_sdpa_masked_batched(query, key, value, mask):
    batch_size, seq_length = query.shape[:2]
    num_attention_heads = 16
    hidden_dim = 1024
    attention_head_size = hidden_dim // num_attention_heads

    new_q_shape = query.size()[:-1] + (num_attention_heads, attention_head_size)
    query = query.view(new_q_shape).permute(0, 2, 1, 3)
    new_k_shape = key.size()[:-1] + (num_attention_heads, attention_head_size)
    key = key.view(new_k_shape).permute(0, 2, 1, 3)
    new_v_shape = value.size()[:-1] + (num_attention_heads, attention_head_size)
    value = value.view(new_v_shape).permute(0, 2, 1, 3)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores /= math.sqrt(attention_head_size)
    target_length = attention_scores.shape[3]

    ## add the causal mask
    if seq_length > 1:
        target_length = attention_scores.shape[3]
        attention_bias = torch.zeros(seq_length, target_length, dtype=query.dtype)
        mask = torch.ones(seq_length, target_length, dtype=torch.bool).tril(diagonal=0)
        attention_bias.masked_fill_(mask.logical_not(), float("-inf"))
        attention_scores += attention_bias

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_probs, value)
    attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, hidden_dim)
    return (attention_output, attention_scores)


def dump_opt_parallel(model, inputs, save, verbose, dropout, home, num_accel):
    model = model.to("cpu")
    inputs = inputs.to("cpu")

    if save:
        if not exists(home):
            makedirs(home)

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
        model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)

    ## iterate over decoder
    for l in range(24):
        block = model.model.decoder.layers[l]

        '''
        OPTForCausalLM(
            (model): OPTModel(
                (decoder): OPTDecoder(
                    (embed_tokens): Embedding(50272, 512, padding_idx=1)
                    (embed_positions): OPTLearnedPositionalEmbedding(2050, 1024)
                    (project_out): Linear(in_features=1024, out_features=512, bias=False)
                    (project_in): Linear(in_features=512, out_features=1024, bias=False)
                    (layers): ModuleList(
                        (0-23): 24 x OPTDecoderLayer(
                            (self_attn): OPTAttention(
                                (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                                (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                                (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                                (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
                                (sdpa): ScratchSDPAMasked()
                            ) + residual
                            (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                            (activation_fn): ReLU()
                            (fc2): Linear(in_features=4096, out_features=1024, bias=True) + residual
                            (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                        )
                    )
                )
            )
            (lm_head): Linear(in_features=512, out_features=50272, bias=False)
        )
        '''
        prefix = f"model.decoder.layers.{l}"

        ## input to self-attention block of decoder
        attn_in = activations[prefix]["input"][0]
        # print(attn_in)
        # exit()
        
        ## query weights, bias, model outputs, local outputs
        query_w = block.self_attn.q_proj.weight
        query_b = block.self_attn.q_proj.bias
        query_o = activations[f"{prefix}.self_attn.q_proj"]["output"][0]
        if len(query_o.shape) < 3:
            ## batch it
            query_o = query_o.unsqueeze(0)
        sw_query_o = torch.einsum("ijk,lk->ijl", attn_in, query_w) + query_b
        split_query_w = split(query_w, num_accel)

        ## key weights, bias, model outputs, local outputs
        key_w = block.self_attn.k_proj.weight
        key_b = block.self_attn.k_proj.bias
        key_o = activations[f"{prefix}.self_attn.k_proj"]["output"][0]
        if len(key_o.shape) < 3:
            ## batch it
            key_o = key_o.unsqueeze(0)
        sw_key_o = torch.einsum("ijk,lk->ijl", attn_in, key_w) + key_b
        split_key_w = split(key_w, num_accel)

        ## value weights, bias, model outputs, local outputs
        value_w = block.self_attn.v_proj.weight
        value_b = block.self_attn.v_proj.bias
        value_o = activations[f"{prefix}.self_attn.v_proj"]["output"][0]
        if len(value_o.shape) < 3:
            ## batch it
            value_o = value_o.unsqueeze(0)
        sw_value_o = torch.einsum("ijk,lk->ijl", attn_in, value_w) + value_b
        split_value_w = split(value_w, num_accel)
        
        ## self-attention outputs and local scratch SDPA attention results
        attn_o = activations[prefix + '.self_attn']["output"][0]
        sw_attn_o, sw_scores_o = scratch_sdpa_masked_batched(sw_query_o, sw_key_o, sw_value_o, inputs["attention_mask"])

        ## first output network
        # dense_in = activations[f"encoder.encoder.layer.{l}.attention.output.dense"]["input"][0]
        dense_w = block.self_attn.out_proj.weight
        dense_b = block.self_attn.out_proj.bias
        dense_o = activations[f"{prefix}.self_attn.out_proj"]["output"][0]
        # if len(dense_o.shape) < 3:
        #     ## batch it
        #     dense_o = dense_o.unsqueeze(0)
        sw_dense_o = torch.einsum("ijk,lk->ijl", sw_attn_o, dense_w) + dense_b
        split_dense_w = split(dense_w, num_accel)
        
        ## first residual and layernorm
        rsd1_o = dense_o + attn_in
        sw_rsd1_o = sw_dense_o + attn_in
        layernorm1_o = activations[f"{prefix}.self_attn_layer_norm"]["output"][0]
        # if len(layernorm1_o.shape) < 3:
        #     ## batch it
        #     layernorm1_o = layernorm1_o.unsqueeze(0)
        layernorm1_w = block.self_attn_layer_norm.weight
        layernorm1_b = block.self_attn_layer_norm.bias

        ## first linear layer
        ffn1_w = block.fc1.weight
        ffn1_b = block.fc1.bias
        relu_in = activations[f"{prefix}.activation_fn"]["input"][0]
        sw_relu_in = torch.einsum("jk,lk->jl", layernorm1_o, ffn1_w) + ffn1_b
        ffn1_o_relu = activations[f"{prefix}.activation_fn"]["output"][0]
        # if len(ffn1_o_relu.shape) < 3:
        #     ## batch it
        #     ffn1_o_relu = ffn1_o_relu.unsqueeze(0)
        sw_ffn1_o_relu = torch.nn.functional.relu(torch.einsum("jk,lk->jl", layernorm1_o, ffn1_w) + ffn1_b)
        split_ffn1_w = split(ffn1_w, num_accel)

        ## second linear layer
        ffn2_w = block.fc2.weight
        ffn2_b = block.fc2.bias
        ffn2_o = activations[f"{prefix}.fc2"]["output"][0]
        # if len(ffn2_o.shape) < 3:
        #     ## batch it
        #     ffn2_o = ffn2_o.unsqueeze(0)
        sw_ffn2_o = torch.einsum("jk,lk->jl", sw_ffn1_o_relu, ffn2_w) + ffn2_b
        split_ffn2_w = split(ffn2_w, num_accel)

        ## second residual and layernorm
        rsd2_o = ffn2_o + layernorm1_o    
        sw_rsd2_o = sw_ffn2_o + layernorm1_o     
        layernorm2_o = activations[f"{prefix}.final_layer_norm"]["output"][0]
        # if len(layernorm2_o.shape) < 3:
        #     ## batch it
        #     layernorm2_o = layernorm2_o.unsqueeze(0)
        layernorm2_w = block.final_layer_norm.weight
        layernorm2_b = block.final_layer_norm.bias

        ## verify output parallelism
        split_query_o = split_mat_mul_batched(attn_in, query_w, query_b, num_accel)
        split_key_o = split_mat_mul_batched(attn_in, key_w, key_b, num_accel)
        split_value_o = split_mat_mul_batched(attn_in, value_w, value_b, num_accel)
        split_dense_o = split_mat_mul_batched(sw_attn_o, dense_w, dense_b, num_accel)
        split_ffn1_o_relu = torch.nn.functional.relu(split_mat_mul_unbatched(layernorm1_o, ffn1_w, ffn1_b, num_accel))
        split_ffn2_o = split_mat_mul_unbatched(sw_ffn1_o_relu, ffn2_w, ffn2_b, num_accel)
        query_verif = compare_tensors(sw_query_o, split_query_o, 1e-4)
        key_verif = compare_tensors(sw_key_o, split_key_o, 1e-4)
        value_verif = compare_tensors(sw_value_o, split_value_o, 1e-4)
        dense_verif = compare_tensors(sw_dense_o, split_dense_o, 1e-4)
        ffn1_verif = compare_tensors(sw_ffn1_o_relu, split_ffn1_o_relu, 1e-4)
        ffn2_verif = compare_tensors(sw_ffn2_o, split_ffn2_o, 1e-4)

        if verbose:
            print(f'---- START DECODER BLOCK {l} ----')
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
            print(f'ffn1_o_relu \t {list(ffn1_o_relu.shape)}')
            print(f'sw_ffn1_o_relu \t {list(sw_ffn1_o_relu.shape)}')
            print(f'output parallelism match: {ffn1_verif}')
            print(f'relu_in \t\t {list(relu_in.shape)}')
            print(f'sw_relu_in \t\t {list(sw_relu_in.shape)}')
            print(f'----> ffn2')
            print(f'ffn2_o \t\t\t {list(ffn2_o.shape)}')
            print(f'sw_ffn2_o \t\t {list(sw_ffn2_o.shape)}')
            print(f'output parallelism match: {ffn2_verif}')
            print(f'----> residual + layernorm 2')
            print(f'rsd2_o \t\t\t {list(rsd2_o.shape)}')
            print(f'sw_rsd2_o \t\t {list(sw_rsd2_o.shape)}')
            print(f'layernorm2_o \t {list(layernorm2_o.shape)}')
            print(f'----  END DECODER BLOCK {l}  ----\n')

        if save:
            np.save(f"{home}/decoder_l{l}_attn_in", attn_in.detach().numpy())
            np.save(f"{home}/decoder_l{l}_query_w", query_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_query_w_t", query_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_query_w", split_query_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_query_b", query_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_query_o", query_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_key_w", key_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_key_w_t", key_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_key_w", split_key_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_key_b", key_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_key_o", key_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_value_w", value_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_value_w_t", value_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_value_w", split_value_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_value_b", value_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_value_o", value_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_attn_o", attn_o.detach().numpy())    
            np.save(f"{home}/decoder_l{l}_sw_scores_o", sw_scores_o.detach().numpy())    
            np.save(f"{home}/decoder_l{l}_dense_w", dense_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_dense_w_t", dense_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_dense_w", split_dense_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_dense_b", dense_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_dense_o", dense_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_sw_rsd1_o", rsd1_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn1_o", layernorm1_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn1_w", layernorm1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn1_b", layernorm1_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_w", ffn1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_w_t", ffn1_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_ffn1_w", split_ffn1_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_b", ffn1_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn1_o", ffn1_o_relu.detach().numpy())
            np.save(f"{home}/decoder_l{l}_relu_in", relu_in.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_w", ffn2_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_w_t", ffn2_w.T.detach().numpy().flatten())
            np.save(f"{home}/decoder_l{l}_split_{num_accel}_ffn2_w", split_ffn2_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_b", ffn2_b.detach().numpy())
            np.save(f"{home}/decoder_l{l}_ffn2_o", ffn2_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_sw_rsd2_o", rsd2_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn2_o", layernorm2_o.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn2_w", layernorm2_w.detach().numpy())
            np.save(f"{home}/decoder_l{l}_lyn2_b", layernorm2_b.detach().numpy())


if __name__ == "__main__":
    model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    # print(model.config)
    # exit()
    inputs = tokenizer([("What are we having for dinner?")], return_tensors="pt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dropout', help='enable dropout layers in model', action='store_true')
    parser.add_argument('-s', '--save', help='save weights, activations, and attention scores', action='store_true')
    parser.add_argument('-v', '--verbose', help='print layer sizes and block numbers', action='store_true')
    parser.add_argument('-a', '--accel', type=int, help='number of accelerators to employ output parallelism', default=1)
    parser.add_argument('-l', '--location', type=str, help='location of home directory for saving', required=True)
    args = parser.parse_args()
    dump_opt_parallel(
        model=model, 
        inputs=inputs, 
        save=args.save, 
        verbose=args.verbose, 
        dropout=args.dropout, 
        home=args.location, 
        num_accel=args.accel
    )
    ## run me!
    ## python3 opt_350m_output_parallel.py -l /example/folder/to/save -a 6 -v -s