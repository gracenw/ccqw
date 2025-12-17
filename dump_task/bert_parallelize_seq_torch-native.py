# flake8: noqa

from torch import nn
import torch
import numpy as np
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForMultipleChoice

# dim_model = 768
# num_head = 12
# len_seq = 512

# def scratch_sdpa(query, key, value, mask):
#     scale = 64 ** -0.5
#     # print("original qkv shape", query.shape)
#     new_q_shape = query.size()[:-1] + (num_head, int(dim_model/num_head))
#     query = query.view(new_q_shape).permute(0, 2, 1, 3)
#     new_k_shape = key.size()[:-1] + (num_head, int(dim_model/num_head))
#     key = key.view(new_k_shape).permute(0, 2, 1, 3)
#     new_v_shape = value.size()[:-1] + (num_head, int(dim_model/num_head))
#     value = value.view(new_v_shape).permute(0, 2, 1, 3)

#     # print("transposed key shape", key.transpose(-1, -2).shape)
#     attention_scores = torch.matmul(query, key.transpose(-1, -2))
#     # print("attention_scores shape", attention_scores.shape)
#     attention_scores = attention_scores*scale

#     attention_output = torch.matmul(attention_scores, value)
#     # print("attention_output shape", attention_output.shape)
#     # print("attention_output permute shape", attention_output.permute(0, 2, 1, 3).shape)
#     attention_output = attention_output.permute(0, 2, 1, 3).reshape(len_seq, dim_model)
#     return attention_output

# def scratch_sdpa_masked(query, key, value, mask, lyr):
#     scale = 64 ** -0.5
#     # print("original qkv shape", query.shape)
#     new_q_shape = query.size()[:-1] + (num_head, int(dim_model/num_head))
#     query = query.view(new_q_shape).permute(0, 2, 1, 3)
#     new_k_shape = key.size()[:-1] + (num_head, int(dim_model/num_head))
#     key = key.view(new_k_shape).permute(0, 2, 1, 3)
#     new_v_shape = value.size()[:-1] + (num_head, int(dim_model/num_head))
#     value = value.view(new_v_shape).permute(0, 2, 1, 3)
#     # print("multi-head qkv shape", new_q_shape)
#     # print("permute qkv shape", query.shape)
    
#     # print("original mask shape", mask.shape)
#     extend_mask = mask.expand(1, len_seq, len_seq).unsqueeze(1).repeat(1, num_head, 1, 1)
#     # print("extended mask shape", mask.expand(1, len_seq, len_seq).shape)
#     boolean_mask = (extend_mask == 0)
#     # print("multi-head extended mask shape", extend_mask.shape)
#     extend_mask = extend_mask.masked_fill_(boolean_mask, -1e9)

#     # print("transposed key shape", key.transpose(-1, -2).shape)
#     attention_scores = torch.matmul(query, key.transpose(-1, -2))
#     # print("attention_scores shape", attention_scores.shape)
#     attention_scores = attention_scores*scale
#     attention_scores += extend_mask

#     np.save(f"./model/bert_base/npy/l{lyr}_attention_scores", attention_scores.detach().numpy())

#     attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#     attention_output = torch.matmul(attention_probs, value)
#     # print("attention_output shape", attention_output.shape)
#     # print("attention_output permute shape", attention_output.permute(0, 2, 1, 3).shape)
#     attention_output = attention_output.permute(0, 2, 1, 3).reshape(len_seq, dim_model)
#     return attention_output

# def fused_sdpa_masked(attn_in, attn_mask, query, key, value_w, value_b, dense_w, dense_b):
#     scale = 64 ** -0.5
#     # print("original qkv shape", query.shape)
#     new_q_shape = query.size()[:-1] + (num_head, int(dim_model/num_head))
#     query = query.view(new_q_shape).permute(0, 2, 1, 3)
#     new_k_shape = key.size()[:-1] + (num_head, int(dim_model/num_head))
#     key = key.view(new_k_shape).permute(0, 2, 1, 3)
#     attention_scores = torch.matmul(query, key.transpose(-1, -2))
#     attention_scores = attention_scores*scale

#     extend_mask = attn_mask.expand(1, len_seq, len_seq).unsqueeze(1).repeat(1, num_head, 1, 1)
#     boolean_mask = (extend_mask == 0)
#     extend_mask = extend_mask.masked_fill_(boolean_mask, -1e9)
#     attention_scores += extend_mask
#     attention_probs = nn.functional.softmax(attention_scores, dim=-1)

#     new_v_shape = (num_head, int(dim_model/num_head)) + value_w.size()[:-1]
#     value_w = value_w.view(new_v_shape)
#     value_b = value_b.view(num_head, int(dim_model/num_head))

#     new_d_shape = dense_w.size()[:-1] + (num_head, int(dim_model/num_head))
#     dense_w = dense_w.view(new_d_shape)

#     # generate (12, 768, 768) and reduce to (768, 768)
#     fused_w = torch.sum(torch.einsum("hdi,ohd->hoi", value_w, dense_w), dim=0)
#     fused_b = torch.sum(torch.einsum("hd,ohd->ho", value_b, dense_w), dim=0)
#     # fused_v = torch.einsum("ijk,lk->ijl", attn_in, fused_w) + fused_b
#     fused_v = torch.einsum("ijk,lk->ijl", attn_in, fused_w)
#     new_fv_shape = fused_v.size()[:-1] + (num_head, int(dim_model/num_head))
#     fused_v = fused_v.view(new_fv_shape).permute(0, 2, 1, 3)
#     attention_output = torch.matmul(attention_probs, fused_v)
#     attention_output = attention_output.permute(0, 2, 1, 3).reshape(len_seq, dim_model)
#     # return attention_output + dense_b
#     return attention_output

# def dump_bert(model, inputs):
#     model = model.to("cpu")
#     inputs = inputs.to("cpu")

#     '''
#     BertForSequenceClassification(
#     (bert): BertModel(
#         (embeddings): BertEmbeddings(
#         (word_embeddings): Embedding(30522, 768, padding_idx=0)
#         (position_embeddings): Embedding(512, 768)
#         (token_type_embeddings): Embedding(2, 768)
#         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#         (dropout): Dropout(p=0, inplace=False)
#         )
#         (encoder): BertEncoder(
#         (layer): ModuleList(
#             (0-11): 12 x BertLayer(
#             (attention): BertAttention(
#                 (self): BertSdpaSelfAttention( ## DIFFERENT
#                 (query): Linear(in_features=768, out_features=768, bias=True)
#                 (key): Linear(in_features=768, out_features=768, bias=True)
#                 (value): Linear(in_features=768, out_features=768, bias=True)
#                 (dropout): Dropout(p=0, inplace=False)
#                 )
#                 (output): BertSelfOutput(
#                 (dense): Linear(in_features=768, out_features=768, bias=True)
#                 (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#                 (dropout): Dropout(p=0, inplace=False) ## DIFFERENT
#                 )
#             )
#             (intermediate): BertIntermediate(
#                 (dense): Linear(in_features=768, out_features=3072, bias=True)
#                 (intermediate_act_fn): GELUActivation()
#             )
#             (output): BertOutput(
#                 (dense): Linear(in_features=3072, out_features=768, bias=True)
#                 (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#                 (dropout): Dropout(p=0, inplace=False) ## DIFFERENT
#             )
#             )
#         )
#         )
#         (pooler): BertPooler(
#         (dense): Linear(in_features=768, out_features=768, bias=True)
#         (activation): Tanh()
#         )
#     )
#     (dropout): Dropout(p=0, inplace=False)
#     (classifier): Linear(in_features=768, out_features=2, bias=True)
#     )
#     '''

#     def register_hooks(model):
#         activations = {}

#         def get_activation(name):
#             def hook(model, input, output):
#                 activations[name] = {'input': input, 'output': output}
#             return hook

#         for name, module in model.named_modules():
#             module.register_forward_hook(get_activation(name))

#         return activations

#     # dump to onnx
#     # return F.scaled_dot_product_attention(..., dropout_p=(self.p if self.training else 0.0))
#     # torch.onnx.export(model, inputs["input_ids"], "./bert/bert.onnx")
    
#     # dump parameter and intermediate to npy
#     activations = register_hooks(model)
#     with torch.no_grad():
#         model(**inputs)

#     # for name, tensors in activations.items():
#     #     if ("embeddings" not in name) and ("." in name) and (name != ""):
#     #         print(f"Layer: {name}")
#             # i_npy_path = "./bert/" + name + ".input"
#             # np.save(i_npy_path, tensors['input'][0].detach().numpy())
#             # o_npy_path = "./bert/" + name + ".output"
#             # np.save(o_npy_path, tensors['output'][0].detach().numpy())
#         # elif name == "bert.embeddings.LayerNorm":
#             # np.save("./bert/bert.encoder.input",  tensors['output'][0].detach().numpy())                    

#     for l in range(len(model.bert.encoder.layer)):
#         block = model.bert.encoder.layer[l]
#     # generating qkv
#         sw_attn_in = activations[f"bert.encoder.layer.{l}.attention"]["input"][0]
#         query_w = block.attention.self.query.weight
#         query_b = block.attention.self.query.bias
#         query_o = activations[f"bert.encoder.layer.{l}.attention.self.query"]["output"][0]
#         key_w = block.attention.self.key.weight
#         key_b = block.attention.self.key.bias
#         key_o = activations[f"bert.encoder.layer.{l}.attention.self.key"]["output"][0]
#         value_w = block.attention.self.value.weight
#         value_b = block.attention.self.value.bias
#         value_o = activations[f"bert.encoder.layer.{l}.attention.self.value"]["output"][0]
#         np.save(f"./model/bert_base/npy/l{l}_in", activations[f"bert.encoder.layer.{l}.attention"]["input"][0].detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_query_w", query_w.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_query_w_t", query_w.T.detach().numpy().flatten())
#         np.save(f"./model/bert_base/npy/l{l}_query_b", query_b.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_query_o", query_o.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_key_w", key_w.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_key_w_t", key_w.T.detach().numpy().flatten())
#         np.save(f"./model/bert_base/npy/l{l}_key_b", key_b.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_key_o", key_o.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_value_w", value_w.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_value_w_t", value_w.T.detach().numpy().flatten())
#         np.save(f"./model/bert_base/npy/l{l}_value_b", value_b.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_value_o", value_o.detach().numpy())
        
#         print("block_in shape", sw_attn_in.shape)
#         print("query_w shape", query_w.shape)
        
#     # calculating attention
#     # should be same as np.load("/home/je/Desktop/2024_rampc/pytorch/bert/bert.encoder.layer.0.attention.output.input.npy")
#         # sw_query_o = torch.einsum("ijk,lk->ijl", sw_attn_in, query_w)
#         # sw_key_o = torch.einsum("ijk,lk->ijl", sw_attn_in, key_w)
#         # sw_value_o = torch.einsum("ijk,lk->ijl", sw_attn_in, value_w)
#         sw_query_o = torch.einsum("ijk,lk->ijl", sw_attn_in, query_w) + query_b
#         sw_key_o = torch.einsum("ijk,lk->ijl", sw_attn_in, key_w) + key_b
#         sw_value_o = torch.einsum("ijk,lk->ijl", sw_attn_in, value_w) + value_b
#         sw_attn_out = scratch_sdpa_masked(sw_query_o, sw_key_o, sw_value_o, inputs["attention_mask"], l)
#         np.save(f"./model/bert_base/npy/l{l}_attn_o", activations[f"bert.encoder.layer.{l}.attention.output"]["input"][0].detach().numpy())

#         dense_w = block.attention.output.dense.weight
#         dense_b = block.attention.output.dense.bias
#         dense_o = activations[f"bert.encoder.layer.{l}.attention.output.dense"]["output"][0]
#         np.save(f"./model/bert_base/npy/l{l}_dense_w", dense_w.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_dense_w_t", dense_w.T.detach().numpy().flatten())
#         np.save(f"./model/bert_base/npy/l{l}_dense_b", dense_b.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_dense_o", dense_o.detach().numpy())
        
#         sw_rsd1_o = dense_o + sw_attn_in                                                                    # input to the layernorm
#         np.save(f"./model/bert_base/npy/l{l}_rsd1_o", sw_rsd1_o.detach().numpy())
#         layernorm_o = activations[f"bert.encoder.layer.{l}.attention.output.LayerNorm"]["output"][0]        # input to the ffn1
#         np.save(f"./model/bert_base/npy/l{l}_lyn1_o", layernorm_o.detach().numpy())
#         print("rsd1 shape", sw_rsd1_o.shape)

#     # FFN
#         ffn1_w = block.intermediate.dense.weight
#         ffn1_b = block.intermediate.dense.bias
#         # ffn1_o = activations[f"bert.encoder.layer.{l}.intermediate.dense"]["output"][0]
#         ffn1_o = activations[f"bert.encoder.layer.{l}.intermediate.intermediate_act_fn"]["output"][0]       # including gelu
#         np.save(f"./model/bert_base/npy/l{l}_ffn1_w", ffn1_w.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_ffn1_w_t", ffn1_w.T.detach().numpy().flatten())
#         np.save(f"./model/bert_base/npy/l{l}_ffn1_b", ffn1_b.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_ffn1_o", ffn1_o.detach().numpy())
#         gelu_in = torch.einsum("jk,lk->jl", layernorm_o, ffn1_w)
#         np.save(f"./model/bert_base/npy/l{l}_gelu", gelu_in.detach().numpy())
#         sw_ffn1_o = torch.nn.functional.gelu(torch.einsum("jk,lk->jl", layernorm_o, ffn1_w) + ffn1_b)
#         print("ffn1 shape", sw_ffn1_o.shape)

#         ffn2_w = block.output.dense.weight
#         ffn2_b = block.output.dense.bias
#         ffn2_o = activations[f"bert.encoder.layer.{l}.output.dense"]["output"][0]
#         np.save(f"./model/bert_base/npy/l{l}_ffn2_w", ffn2_w.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_ffn2_w_t", ffn2_w.T.detach().numpy().flatten())
#         np.save(f"./model/bert_base/npy/l{l}_ffn2_b", ffn2_b.detach().numpy())
#         np.save(f"./model/bert_base/npy/l{l}_ffn2_o", ffn2_o.detach().numpy())
#         sw_ffn2_o = torch.einsum("jk,lk->jl", sw_ffn1_o, ffn2_w) + ffn2_b
#         print("ffn2 shape", sw_ffn2_o.shape)

#         sw_rsd2_o = ffn2_o + layernorm_o     
#         np.save(f"./model/bert_base/npy/l{l}_rsd2_o", sw_rsd2_o.detach().numpy())                           # input to the layernorm
#         layernorm2_o = activations[f"bert.encoder.layer.{l}.output.LayerNorm"]["output"][0]                 # input to the nex block
#         np.save(f"./model/bert_base/npy/l{l}_lyn2_o", layernorm2_o.detach().numpy())
#         print("rsd2 shape", sw_rsd2_o.shape)
#         print()


def main():
    # print(torch.__version__)
    # config = BertConfig(hidden_dropout_prob=0)
    device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")
    config = BertConfig(
        hidden_dropout_prob=0, 
        _attn_implementation = 'sdpa',
        is_decoder=True,
        add_cross_attention=True,
    )

    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", config=config).to(device)
    inputs = tokenizer("my cat is the cutest cat ever!", return_tensors="pt").to(device)
    outputs = model(**inputs)
    # print(model(**inputs))
    print(outputs.logits.argmax().item())

    inputs = tokenizer("i do not like running outside in the wind and cold.", return_tensors='pt').to(device)
    outputs = model(**inputs)
    print(outputs.logits.argmax().item())
    exit()
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForMultipleChoice.from_pretrained('bert-base-uncased', config=config).to(device)

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."

    labels = torch.tensor(0).unsqueeze(0).to(device) # choice0 is correct (according to Wikipedia ;)), batch size 1

    encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
    inputs = {k: v.unsqueeze(0).to(device) for k,v in encoding.items()}
    outputs = model(**inputs, labels=labels) # batch size is 1

    _, logits = outputs[:2]
    predicted_class = logits.argmax().item()
    print(predicted_class)

    # dump_bert(model, inputs)


if __name__ == '__main__':
    main()