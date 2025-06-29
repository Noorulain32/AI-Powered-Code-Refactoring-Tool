from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import pandas as pd
import torch
import os

output_dir = os.path.join("model", "codeT5_finetuned")
os.makedirs(output_dir, exist_ok=True)

csv_path = "code_refinement_dataset.csv"
df = pd.read_csv(csv_path)
dataset = Dataset.from_pandas(df[['text', 'code']])

model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(example):
    try:
        inputs = tokenizer(
            f"refactor Python: {example['code']}",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        targets = tokenizer(
            example['code'],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        inputs["labels"] = targets["input_ids"]
        return inputs
    except Exception as e:
        print(f"❌ Error during tokenization: {e}")
        return None

print("🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, batched=False)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    log_level='debug',
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

print("💾 Testing model save before training...")
try:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Model save test passed.")
except Exception as e:
    print(f"❌ Error during pre-training save: {e}")

print("🚀 Starting training...")
try:
    trainer.train()
    print("✅ Training finished. Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Fine-tuning complete. Model saved to: {output_dir}")
except Exception as e:
    print(f"❌ Error during training or saving: {e}")
