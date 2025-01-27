import os
from openai import OpenAI


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)
# Define the caption
caption = "A brown dog is running on a grassy field with a ball in its mouth."

# Create a prompt to instruct GPT-4o to extract objects
prompt = f"""
Extract the objects mentioned in the following caption:

Caption: "{caption}"

List only the objects found in the caption as bullet points.
"""

# Use the OpenAI GPT-4 API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an assistant that extracts objects from text."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,  # Ensure consistent responses
)

# Extract and display the objects
objects = response['choices'][0]['message']['content']
print("Extracted objects:")
print(objects)