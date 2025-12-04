import torch
from torch import nn
import numpy as np
from models import BertForSequenceClassification, EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained(
    "patrickvonplaten/bert2bert_cnn_daily_mail"
)

print(dir(nn.LayerNorm(768, 0.005)))

for name, param in model.named_parameters():
    if 'LayerNorm' in name:
        print(name)
        print(list(param.shape))
        np.save(f"/home/gracen/repos/rampc/model/bert_base/encoder_l{l}_attn_o", attn_o.detach().numpy())



# decoder.bert.encoder.layer.11.attention.output.LayerNorm.weight
# [768]
# decoder.bert.encoder.layer.11.attention.output.LayerNorm.bias
# [768]
# decoder.bert.encoder.layer.11.crossattention.output.LayerNorm.weight
# [768]
# decoder.bert.encoder.layer.11.crossattention.output.LayerNorm.bias
# [768]
# decoder.bert.encoder.layer.11.output.LayerNorm.weight
# [768]
# decoder.bert.encoder.layer.11.output.LayerNorm.bias
# [768]

# [768]
# encoder.encoder.layer.4.attention.output.LayerNorm.weight
# [768]
# encoder.encoder.layer.4.attention.output.LayerNorm.bias
# [768]
# encoder.encoder.layer.4.output.LayerNorm.weight
# [768]
# encoder.encoder.layer.4.output.LayerNorm.bias
# [768]