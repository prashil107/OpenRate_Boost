import pandas as pd

input_file = 'emails/emails.csv'   
output_file = 'emails/emails_cleaned.csv'

chunksize = 10000  # Adjust based on your memory capacity

def is_forwarded(subject):
    if isinstance(subject, str):
        return subject.lower().startswith(('fw:', 'fwd:', 'fw -', 'fwd -'))
    return False

def is_automated_reply(body):
    if isinstance(body, str):
        keywords = ['auto reply', 'automatic reply', 'out of office', 'do not reply', 'automatic response', 'no-reply']
        body_lower = body.lower()
        return any(keyword in body_lower for keyword in keywords)
    return False

write_header = True  # Write header only for first chunk

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    # Normalize column names to lower case and strip spaces if needed
    chunk.columns = chunk.columns.str.strip().str.lower()

    # Filter out forwarded emails using 'file' column as subject
    chunk = chunk[~chunk['file'].apply(is_forwarded)]

    # Filter out automated replies using 'message' column as body
    chunk = chunk[~chunk['message'].apply(is_automated_reply)]

    # Drop rows with missing 'file' or 'message'
    chunk = chunk.dropna(subset=['file', 'message'])

    # Filter out very short content to remove noise
    chunk = chunk[(chunk['file'].str.len() > 5) & (chunk['message'].str.len() > 20)]

    # Append cleaned chunk to output CSV
    chunk.to_csv(output_file, mode='w' if write_header else 'a', index=False, header=write_header)
    write_header = False

print(f"Cleaning complete. Cleaned data saved to: {output_file}")
