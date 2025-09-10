OpenRate Boost: Fine-Tune GPT-2 for Email Subject Generation


A Python project to fine-tune GPT-2 for generating email subject lines from Enron-style datasets using Hugging Face Transformers.


Project Motivation


This project helps automate the generation of concise email subject lines from long email bodies using fine-tuned GPT-2. It is useful for improving email workflows, productivity apps, or text summarization research.

Features

1.Prepare large email datasets for NLP.

2.Fine-tune GPT-2 on custom email body/subject data.

3.Generate subject lines from new email text.

4.Step-by-step scripts and troubleshooting guides.

Technologies: Python 3.12+, transformers (Hugging Face), pandas,torch,accelerate, datasets


Dataset Preparation


1.Place your raw dataset (e.g., emails.csv) in the project folder.

2.Convert CSV into plain text files for training/testing using:

   Each line: [body text] [SEP] [subject line]

   Use scripts provided (see data_prep.py for details).

Installation

    python -m pip install --upgrade transformers torch accelerate pandas datasets
Usage


Step-by-step Workflow

1. Prepare Data
   ` python
    import pandas as pd
    df = pd.read_csv("emails.csv")
    df_clean = df.dropna(subset=["file", "message"]).sample(frac=1, random_state=42)
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split_idx]
    df_test = df_clean.iloc[split_idx:]
    def format_line(row):
        return f"{row['message'].strip()} [SEP] {row['file'].strip()}"
    with open("train.txt", "w", encoding="utf-8") as trainf:
        for _, row in df_train.iterrows(): trainf.write(format_line(row) + "\n")
    with open("test.txt", "w", encoding="utf-8") as testf:
        for _, row in df_test.iterrows(): testf.write(format_line(row) + "\n")`


2. Fine-tune GPT-2
`python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("text", data_files={"train": "train.txt", "test": "test.txt"})

def tokenize_fn(examples): return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
train_ds = dataset["train"].map(tokenize_fn, batched=True)
test_ds = dataset["test"].map(tokenize_fn, batched=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./gpt2_enron_subjects",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("./gpt2_enron_subjects")
tokenizer.save_pretrained("./gpt2_enron_subjects")```


3. Generate Subject Lines
`python
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2_enron_subjects", tokenizer="./gpt2_enron_subjects")
prompt = "Please review the attached report by EOD. [SEP]"
result = generator(prompt, max_length=40, num_return_sequences=1)
print(result[0]["generated_text"])`
Hardware Recommendations
Recommended: NVIDIA RTX 3060 or better for fast training.

Minimum: Any CUDA-enabled GPU with 4GB+ VRAM, or risk much slower training on CPU.

Large datasets (500k+ samples) may require long training times; consider batch adjustments and subset experiments.

Troubleshooting
TypeError: unexpected keyword argument 'evaluation_strategy'
Solution: Upgrade transformers with python -m pip install --upgrade transformers.

ImportError: requires accelerate>=0.26.0
Solution: Install via pip install --upgrade accelerate.

Script slow or seems stuck?
Large data, small batch size, and CPU-only all slow down training. Print logs and test on a dataset sample.

License
This project is freely available under the MIT License.
