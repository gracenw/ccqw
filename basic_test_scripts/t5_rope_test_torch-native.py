# flake8: noqa
import torch, evaluate
import numpy as np
from models.modeling_t5_local_rope import T5RoPEForConditionalGeneration
from transformers import T5Tokenizer #, T5ForConditionalGeneration



def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def main():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=False)
    model = T5RoPEForConditionalGeneration.from_pretrained(
        # '/home/gracen/repos/rampc/models/weights/t5_basolute/version_000'
        'google-t5/t5-base'
    )
    model.eval()
    print(model)

    input_ids = tokenizer(
        "translate English to German: I love my cat very much!", return_tensors="pt"
    ).input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()