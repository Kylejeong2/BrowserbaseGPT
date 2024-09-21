import json
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

def extract_text(item: Dict) -> str:
    """
    Extracts and concatenates text content from the item.
    """
    content_parts = []
    def recurse_extract(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                recurse_extract(value)
        elif isinstance(obj, list):
            for element in obj:
                recurse_extract(element)
        elif isinstance(obj, str):
            content_parts.append(obj)
    recurse_extract(item)
    return '\n'.join(content_parts)

def generate_prompt_completion(content: str, num_pairs: int = 100) -> List[Dict[str, str]]:
    """
    Uses GPT-4o-mini to generate multiple question and answer pairs from the content.
    """
    gpt_prompt = f"""Based on the following documentation content, generate {num_pairs} diverse question and answer pairs. Ensure the questions cover different aspects of the content and vary in complexity.

Content:
\"\"\"
{content}
\"\"\"

Generate {num_pairs} Q&A pairs in the following format:
Q1: [Question]
A1: [Answer]

Q2: [Question]
A2: [Answer]

... and so on.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates question and answer pairs based on given content."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=2000,
            n=1,
            temperature=0.7,
        )
        
        text = response.choices[0].message.content.strip()
        pairs = []
        for pair in text.split('\n\n'):
            if pair.startswith('Q') and 'A' in pair:
                question, answer = pair.split('\n', 1)
                question = question.split(':', 1)[1].strip()
                answer = answer.split(':', 1)[1].strip()
                pairs.append({'prompt': "Q: " + question + "\nA:", 'completion': " " + answer})
        
        return pairs
    except Exception as e:
        print(f"Error generating prompt-completion pairs: {e}")
        return []

def main():
    # Load the scraped data
    with open('scraping_browserbase_scripts/scraped_docs.json', 'r') as file:
        data = json.load(file)

    prompt_completion_list = []
    for item in data:
        # Extract relevant text content
        content = extract_text(item)
        if not content.strip():
            continue  # Skip if content is empty

        # Generate multiple prompt-completion pairs using GPT-4
        pairs = generate_prompt_completion(content)
        prompt_completion_list.extend(pairs)

    # Write the prompt-completion pairs to a JSONL file
    with open('training_data.jsonl', 'w') as outfile:
        for entry in prompt_completion_list:
            json_line = json.dumps(entry, ensure_ascii=False)
            outfile.write(json_line + '\n')

    print(f"Successfully wrote {len(prompt_completion_list)} entries to training_data.jsonl.")

if __name__ == "__main__":
    main()