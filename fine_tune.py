import os
import json
import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# データの読み込み
data_path = os.path.join(BASE_DIR, "data", "data.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Dataset形式に変換
dataset = Dataset.from_list(data)

# モデルの指定
model_name = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# トークン化
def tokenize_function(examples):
    inputs = [f"{x}" for x in examples["input_text"]]
    targets = examples["output_text"]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(text_target=targets, padding="max_length", truncation=True, max_length=256)  
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# データの前処理
tokenized_datasets = dataset.map(tokenize_function, remove_columns=["input_text", "output_text"], batched=True)
split_data = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True)
train_dataset = split_data["train"].with_format("torch")
eval_dataset = split_data["test"].with_format("torch")

# トレーニング設定
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    gradient_accumulation_steps=4,  
    max_grad_norm=0.5,  
    learning_rate=3e-5,
    warmup_steps=2000,
    weight_decay=0.01,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",  
    logging_dir=os.path.join(BASE_DIR, "logs"),
    logging_steps=1000,
    save_steps=1000,
    eval_steps=1000,
    load_best_model_at_end=True,
    lr_scheduler_type="cosine"
)

# Trainerの定義
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# ファインチューニングの実行
trainer.train()

# モデルの保存
model_save_path = os.path.join(BASE_DIR, "trained_model")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("モデルのファインチューニングが完了しました！")