from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, GPT2ForQuestionAnswering

import torch

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2ForQuestionAnswering.from_pretrained("openai-community/gpt2")

    question, text = "Are cats cute?", "Yes, cats are very cute"

    inputs = tokenizer(question, text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        print(outputs)


if __name__ == '__main__':
    main()
