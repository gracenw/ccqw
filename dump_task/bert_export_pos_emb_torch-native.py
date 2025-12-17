# flake8: noqa

from transformers import BertTokenizerFast
from models import EncoderDecoderModel
import torch
import numpy as np


if __name__ == '__main__':
    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
    print(model)
    enc_pos_emb = model.encoder.embeddings.position_embeddings.weight
    dec_pos_emb = model.decoder.bert.embeddings.position_embeddings.weight
    np.save(f"/home/gracen/repos/rampc/models/kernels/enc_pos_emb", enc_pos_emb.detach().numpy())
    np.save(f"/home/gracen/repos/rampc/models/kernels/dec_pos_emb", dec_pos_emb.detach().numpy())