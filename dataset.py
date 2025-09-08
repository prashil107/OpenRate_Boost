import pandas as pd

# Load the CSV dataset
df = pd.read_csv("emails/emails.csv")  # Use your actual filename/path

# Drop rows with missing data in either 'file' or 'message'
df_clean = df.dropna(subset=["file", "message"])

# Shuffle the DataFrame
df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# Split 80% train / 20% test
train_frac = 0.8
train_size = int(len(df_shuffled) * train_frac)

df_train = df_shuffled.iloc[:train_size]
df_test = df_shuffled.iloc[train_size:]

# Format rows as 'body [SEP] subject'
def format_line(row):
    return f"{row['message'].strip()} [SEP] {row['file'].strip()}"

# Write to train.txt
with open("train.txt", "w", encoding="utf-8") as trainfile:
    for _, row in df_train.iterrows():
        trainfile.write(format_line(row) + "\n")

# Write to test.txt
with open("test.txt", "w", encoding="utf-8") as testfile:
    for _, row in df_test.iterrows():
        testfile.write(format_line(row) + "\n")

print(f"Saved {len(df_train)} lines in train.txt and {len(df_test)} lines in test.txt.")
