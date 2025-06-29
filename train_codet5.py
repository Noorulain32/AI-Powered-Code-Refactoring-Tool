import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# Step 1: Load and rename columns
df = pd.read_csv("train_data_python.csv")
df = df.rename(columns={"buggy_python": "source", "fixed_python": "target"})

# Optional: Remove rows with missing values
df = df.dropna(subset=["source", "target"])

# Step 2: Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Step 3: Split dataset into train and validation (10% for validation)
split = dataset.train_test_split(test_size=0.1)
train_dataset = split['train']
eval_dataset = split['test']

# Step 4: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

# Step 5: Tokenize inputs
def tokenize_function(example):
    inputs = tokenizer(example["source"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(example["target"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Step 6: Load model
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# Step 7: Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,  
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
)

# Step 8: Initialize Trainer with eval_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Step 9: Start training
trainer.train()
# Step 9: Save the trained model to a directory
trainer.save_model("final_codet5_model")
print("Model saved to 'final_codet5_model' directory.")
