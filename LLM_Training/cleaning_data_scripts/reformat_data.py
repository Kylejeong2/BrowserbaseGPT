import json

def reformat_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            
            messages = [
                {"role": "system", "content": "You are a technical expert on BrowserBase, focusing on its Rest API and coding aspects. You're a master at Python and Javascript/Node.js."},
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["completion"]}
            ]
            
            output = {"messages": messages}
            json.dump(output, outfile, ensure_ascii=False)
            outfile.write('\n')

# Usage
input_file = './training_data.jsonL'
output_file = './training_data_reformatted.jsonl'
reformat_data(input_file, output_file)