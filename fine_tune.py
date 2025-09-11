import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from datasets import load_dataset

# Tokenize the Data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Add [SEP] as a special token if needed
special_tokens_dict = {'additional_special_tokens': ['[SEP]']}
tokenizer.add_special_tokens(special_tokens_dict)

dataset = load_dataset("text", data_files={"train": "train.txt", "test": "test.txt"})

def preprocess(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)  # Reduce max_length
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Only tokenize train and test once
tokenized_train = dataset["train"].map(preprocess, batched=True)
tokenized_test = dataset["test"].map(preprocess, batched=True)

# Load Pretrained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Adjust for special tokens

# Setup Training Arguments and Trainer
os.makedirs("./gpt2_enron_subjects", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
training_args = TrainingArguments(
    output_dir="./gpt2_enron_subjects",
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    fp16=True,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    seed=42,  # Set your desired seed value here
)

model.config.pad_token_id = tokenizer.pad_token_id

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    #eval_dataset=tokenized_test,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./gpt2_enron_subjects")
tokenizer.save_pretrained("./gpt2_enron_subjects")
print("Training completed and model saved.")

# Generate email subject lines example (no need to reload dataset)
loaded_model = GPT2LMHeadModel.from_pretrained("./gpt2_enron_subjects")
loaded_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_enron_subjects")
generator = pipeline("text-generation", model=loaded_model, tokenizer=loaded_tokenizer)

prompt = "Please review the attached report by EOD. [SEP]"
output = generator(prompt, max_length=40, num_return_sequences=1)

print("Generated Subject:", output[0]["generated_text"])
