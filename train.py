from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

dataset = load_dataset('json', data_files='data.json', split='train')
dataset = dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("chiraglol/r-igcse-reputation-quality-predictor")

def tokenize_function(examples): 
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "chiraglol/r-igcse-reputation-quality-predictor", 
    num_labels=1 
)

model.config.problem_type = "regression"

training_args = TrainingArguments(
    output_dir="./output",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Starting training...")
trainer.train()

trainer.save_model("./output/final")
print("Model saved to ./output/final")