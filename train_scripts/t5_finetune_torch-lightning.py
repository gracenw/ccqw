# train.py
import os
import torch
import datasets
from models import T5ForConditionalGeneration
from transformers import T5Tokenizer
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


class LitTextSummarization(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base").train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        # for name, param in self.model.named_parameters():
        #     if not 'position_embeddings' in name:
        #         param.requires_grad = False

    def training_step(self, batch):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", output.loss)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-4, weight_decay=0.01)


class TextSummarizationData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=True)

    def prepare_data(self):
        if os.path.isdir("data/cnn_dailymail"): return
        dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")
        dataset = dataset["train"].map(self.preprocess_data, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataset.save_to_disk("data/cnn_dailymail")

    def preprocess_data(self, examples):
        inputs = ["summarize: " + text for text in examples["article"]]
        model_inputs = self.tokenizer(inputs, return_tensors="pt") #, max_length=512, truncation=True, padding="max_length")
        # with self.tokenizer.as_target_tokenizer():
        labels = self.tokenizer(examples["highlights"], return_tensors="pt") #, max_length=150, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_dataloader(self):
        dataset = datasets.load_from_disk("data/cnn_dailymail")
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)


if __name__ == "__main__":
    model = LitTextSummarization()
    data = TextSummarizationData()
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=500
    )
    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data)
