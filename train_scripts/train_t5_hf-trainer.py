from transformers import T5TokenizerFast, T5Config
from modeling_t5_local import T5ForConditionalGeneration
from datasets import load_dataset

from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer


if __name__ == '__main__':
    # tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small", legacy=False)
    tokenizer = T5TokenizerFast.from_pretrained("spanish-t5-tokenizer", legacy=False)
    config = T5Config(
        _attn_implementation='sdpa'
    )
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to('cuda')
    # print(dir(tokenizer))
    # print(tokenizer.special_tokens_map)
    # print(tokenizer.vocab_size)
    # exit()

    dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split='train')

    # tokenizer = ByteLevelBPETokenizer()

    # def batch_iterator(batch_size=1000):
    #     for i in range(0, len(dataset), batch_size):
    #         yield dataset[i: i + batch_size]["text"]

    # # Customized training
    # tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=tokenizer.vocab_size)
    # #                                   , min_frequency=2, special_tokens=[
    # #     "</s>",
    # #     "<pad>",
    # #     "<unk>",
    # # ])

    # # Save files to disk
    # tokenizer.save_pretrained("spanish-t5-tokenizer")

    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False #, mlm_probability=0.15
    )

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="spanish-t5",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    # input_ids = tokenizer(
    #     "translate English to Spanish: I love my cat very much", return_tensors="pt"
    # ).input_ids.to('cuda')
    # outputs = model.generate(input_ids)

    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

    # the forward function automatically creates the correct decoder_input_ids

    # loss = model(input_ids=input_ids, labels=labels).loss

    # loss.item()