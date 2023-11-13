import json
import os



def convert_to_jsonl_string(s):
    # Split the string into lines and filter out any empty lines
    json_objects = [line for line in s.strip().split('\n') if line.strip()]
    # Replace single quotes with double quotes to make it valid JSON
    json_objects = [json.loads(obj.replace("'", "\'")) for obj in json_objects]
    # Convert each JSON object into a string and join with a newline to form JSONL
    return '\n'.join(json.dumps(obj) for obj in json_objects)

def save_as_jsonl_files(jsonl_output: str, folder_name: str='.', file_name:str = 'json_output.jsonl'):
    streamed_jsonl = convert_to_jsonl_string(jsonl_output)
    with open(os.path.join(folder_name, file_name), 'w') as f:
        f.write(streamed_jsonl)