from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from datasets import load_dataset

# Step 3: Tokenize the Data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

dataset = load_dataset("text", data_files={"train": "train.txt", "test": "test.txt"})

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = dataset["train"].map(preprocess, batched=True)
tokenized_test = dataset["test"].map(preprocess, batched=True)


# Step 4: Load Pretrained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Adjust for special tokens


# Step 5: Setup Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./gpt2_enron_subjects",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)


# Step 6: Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./gpt2_enron_subjects")
tokenizer.save_pretrained("./gpt2_enron_subjects")


# Step 7: Generate email subject lines example
generator = pipeline("text-generation", model="./gpt2_enron_subjects", tokenizer="./gpt2_enron_subjects")

prompt = "Please review the attached report by EOD. [SEP]"
output = generator(prompt, max_length=40, num_return_sequences=1)

print("Generated Subject:", output[0]["generated_text"])
