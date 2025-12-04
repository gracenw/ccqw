# flake8: noqa
from transformers import T5Tokenizer, T5ForConditionalGeneration


def main():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    input_ids = tokenizer(
        "translate English to German: I love my cat very much", return_tensors="pt"
    ).input_ids
    outputs = model.generate(input_ids)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()