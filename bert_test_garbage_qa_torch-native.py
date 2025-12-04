from transformers import BertTokenizerFast, BertForQuestionAnswering, BertConfig
import torch, evaluate
from torch.nn import CrossEntropyLoss

bleu = evaluate.load('bleu')

config = BertConfig(
    hidden_dropout_prob=0, 
    hidden_size=1024,
    num_attention_heads=16,
    num_hidden_layers=24,
    intermediate_size=4096
)
tokenizer = BertTokenizerFast.from_pretrained(
    'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad',
)
model = BertForQuestionAnswering.from_pretrained(
    'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad',
    config=config,
)

question, context = "Who was Jim Henson?", "Jim Henson was a nice puppet"

target_start_index = torch.tensor([10])
target_end_index = torch.tensor([12])

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
print(inputs)

for i, token in enumerate(inputs.input_ids[0]):
    print(i, ':', tokenizer.decode([token], skip_special_tokens=True))

loss = outputs.loss
print(loss.item())

# sometimes the start/end positions are outside our model inputs, we ignore these terms
ignored_index = outputs.start_logits.size(1)
start_positions = target_start_index.clamp(0, ignored_index)
end_positions = target_end_index.clamp(0, ignored_index)

criterion = CrossEntropyLoss(ignore_index=ignored_index)
start_loss = criterion(outputs.start_logits, start_positions)
end_loss = criterion(outputs.end_logits, end_positions)
total_loss = (start_loss + end_loss) / 2
print(total_loss)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

print(answer_start_index, answer_end_index)

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
predict_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

print(predict_answer)

results = bleu.compute(predictions=[predict_answer], references=[['nice puppet', 'a nice puppet']], tokenizer=tokenizer)

# print(results)