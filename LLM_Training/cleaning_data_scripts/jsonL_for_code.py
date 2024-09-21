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

def generate_api_coding_qa(content: str, num_pairs: int = 100) -> List[Dict[str, str]]:
    """
    Uses GPT-4o-mini to generate question and answer pairs focused on BrowserBase's REST API and coding aspects.
    """
    gpt_prompt = f"""Based on the following BrowserBase documentation content, generate {num_pairs} diverse question and answer pairs specifically focusing on the REST API, coding examples, and technical implementation details. Ensure the questions cover different aspects of using the API, writing code with BrowserBase, and understanding its technical features.

Content:
\"\"\"
{content}
\"\"\"

Generate {num_pairs} Q&A pairs in the following format:
Q1: [API or coding-related question]
A1: [Detailed technical answer, including code snippets where appropriate]

Q2: [API or coding-related question]
A2: [Detailed technical answer, including code snippets where appropriate]

... and so on.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical expert on BrowserBase, focusing on its Rest API and coding aspects. You're a master at Python and Javascript/Node.js."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=3000,
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
        print(f"Error generating API and coding Q&A pairs: {e}")
        return []

def is_api_coding_content(content: str) -> bool:
    """
    Check if the content is related to API or coding aspects of BrowserBase.
    """
    keywords = ['api', 'endpoint', 'request', 'response', 'json', 'http', 'code', 'function', 
                'method', 'parameter', 'return', 'example', 'snippet', 'implementation']
    return any(keyword in content.lower() for keyword in keywords)

def main():
    # Load the scraped data
    with open('scraping_browserbase_scripts/scraped_docs.json', 'r') as file:
        data = json.load(file)

    api_coding_qa_list = []
    for item in data:
        # Extract relevant text content
        content = extract_text(item)
        if not content.strip() or not is_api_coding_content(content):
            continue  # Skip if content is empty or not API/coding related

        # Generate API and coding focused Q&A pairs
        pairs = generate_api_coding_qa(content)
        api_coding_qa_list.extend(pairs)

    # Write the API and coding Q&A pairs to a JSONL file
    with open('api_coding_training_data.jsonl', 'w') as outfile:
        for entry in api_coding_qa_list:
            json_line = json.dumps(entry, ensure_ascii=False)
            outfile.write(json_line + '\n')

    print(f"Successfully wrote {len(api_coding_qa_list)} API and coding Q&A entries to api_coding_training_data.jsonl.")

if __name__ == "__main__":
    main()