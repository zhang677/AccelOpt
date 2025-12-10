import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("--original_base_prompt_path", type=str, required=True)
args.add_argument("--summarizer_output_list_path", type=str, required=True)
args.add_argument("--new_base_prompt_path", type=str, required=True)
args = args.parse_args()

original_base_prompt_path = args.original_base_prompt_path
summarizer_output_list_path = args.summarizer_output_list_path
new_base_prompt_path = args.new_base_prompt_path

with open(original_base_prompt_path, "r") as f:
    base_prompt = f.read()

with open(summarizer_output_list_path, "r") as f:
    summarizer_output_list = json.load(f)

additional_prompt = ""
for item in summarizer_output_list:
    if item["title"] == "**No optimization found**":
        continue
    additional_prompt += f"{item['summary']}\n\n"

if additional_prompt != "":
    base_prompt += "\n\n#Experiences\n"
    base_prompt += additional_prompt

with open(new_base_prompt_path, "w") as f:
    f.write(base_prompt)