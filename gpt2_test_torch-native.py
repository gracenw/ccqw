from models import GPT2ForQuestionAnswering, GPT2Model
from transformers import GPT2Tokenizer

import torch

def main():
    # tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    # model = GPT2ForQuestionAnswering.from_pretrained("openai-community/gpt2")

    # question, text = "Are cats cute?", "Yes, cats are very cute"

    # inputs = tokenizer(question, text, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     print(outputs)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2Model.from_pretrained('gpt2-medium')
    input_ids = tokenizer(
        "What are some cute names for a female cat?", return_tensors="pt"
    ).input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
