import pandas as pd

# Replace this path with the actual path to your CSV file
file_path = "emails/emails.csv"

df = pd.read_csv(file_path)
print("First 5 rows of dataset:")
print(df.head())
