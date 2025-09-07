import openai

openai.api_key = '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'  # Replace with your OpenAI API key

def generate_subject(email_body):
    prompt = f"""Generate a catchy, concise email subject line for the following email body.
Keep it under 15 characters.

Email body: {email_body}

Subject line:"""

    response = openai.chat.completions.create(
        model="gpt-5",  #model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating email subject lines."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        n=1
    )
    return response.choices[0].message.content.strip()

# Example usage
print(generate_subject("Please review the attached report by EOD."))
