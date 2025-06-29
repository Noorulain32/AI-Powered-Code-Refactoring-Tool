import pandas as pd

# Load the custom dataset
df = pd.read_csv("train_data_python.csv")

# Rename columns for CodeT5 fine-tuning format
df = df.rename(columns={
    "buggy_python": "text",    # 'text' will be the input (buggy code)
    "fixed_python": "code"     # 'code' will be the target (fixed code)
})

# Save only relevant columns
df[["text", "code"]].to_csv("code_refinement_dataset.csv", index=False)

print("✅ Custom Python dataset saved to code_refinement_dataset.csv")
