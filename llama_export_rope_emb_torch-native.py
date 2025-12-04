# flake8: noqa

from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import numpy as np


if __name__ == '__main__':
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    print(model)

    # enc_pos_emb = model.encoder.embeddings.position_embeddings.weight
    # dec_pos_emb = model.decoder.bert.embeddings.position_embeddings.weight
    # np.save(f"/home/gracen/repos/rampc/models/kernels/enc_pos_emb", enc_pos_emb.detach().numpy())
    # np.save(f"/home/gracen/repos/rampc/models/kernels/dec_pos_emb", dec_pos_emb.detach().numpy())