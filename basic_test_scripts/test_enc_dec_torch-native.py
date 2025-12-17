# flake8: noqa

from transformers import BertTokenizerFast
from ..models.modeling_enc_dec_local import EncoderDecoderModel


if __name__ == '__main__':
    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
    tokenizer = BertTokenizerFast.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

    article = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )

    tokenized = tokenizer(article, return_tensors="pt")
    input_ids = tokenized.input_ids
    input_mask = tokenized.attention_mask
    
    generated_ids = model.generate(input_ids, attention_mask=input_mask)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(generated_text)