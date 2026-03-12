import os
import json
import argparse
import torch
import transformers
import outlines
from pydantic import BaseModel, Field
from tqdm import tqdm
import time
import pandas as pd

class BookPlateData(BaseModel):
    plate_number: str = Field(
        description="Number below the body of text, preceded by 'PLATE.' Only include the number."
    )
    common_name: str = Field(
        description="Found at the top of the page in all capital letters, but return this field in lowercase letters"
    )
    scientific_name: str = Field(
        description="Below the common name, the italicized part."
    )
    author: str = Field(
        description="Person who first validly published the botanical name."
    )
    altitude_feet: str = Field(
        default="",
        description="Height in feet where specimen was found. Blank if not listed."
    )
    geographic_range: str = Field(
        description="Geographic terms indicating where species can be found."
    )
    specimen_location: str = Field(
        description="Specific location where specimen was obtained"
    )
    colors_listed: str = Field(
        description="Comma-separated list of colors in order mentioned, excluding plant names"
    )

def format_prompt(tokenizer, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stats_file", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if 'SCRATCH' in os.environ:
        scratch_dir = os.path.join(os.environ['SCRATCH'], 'huggingface')
    else:
        scratch_dir = None

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        cache_dir=scratch_dir
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name, 
        cache_dir=scratch_dir
    )

    model = outlines.from_transformers(hf_model, hf_tokenizer)

    schema_json = json.dumps(BookPlateData.model_json_schema(), indent=2)
    
    instructions_base = (
        "You are an expert at extracting botanical specimen data from book pages. "
        "Extract all requested information accurately from the provided text. "
        "If a specific piece of information is not found in the text, return an empty string "
        "for that field. Do NOT use 'Unknown', 'N/A', or similar placeholders.\n\n"
        "Strictly follow the field descriptions in this schema:\n"
        f"{schema_json}"
    )

    files = [f for f in os.listdir(args.input_dir) if f.endswith('.txt')]
    
    if args.test:
        files = files[:5]

    if not files:
        print(f"No .txt files found in {args.input_dir}")
        return

    llm_stats = []

    for filename in tqdm(files, desc="Processing files", unit="file"):
        input_path = os.path.join(args.input_dir, filename)
        pageid = os.path.splitext(filename)[0]
        output_filename = pageid + ".json"
        output_path = os.path.join(args.output_dir, output_filename)

        try:
            start_time = time.perf_counter()
            with open(input_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

            user_message = f"""{instructions_base}

Extract the botanical data from the following text and return it as JSON.

Text: {text_content}

Botanical Data:"""

            prompt = format_prompt(hf_tokenizer, user_message)

            result = model(
                prompt, 
                BookPlateData, 
                max_new_tokens=512,
                do_sample=False
            )

            book_data = BookPlateData.model_validate_json(result)
            book_dict = book_data.model_dump()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(book_dict, f, indent=2)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            stats = {'pageid': pageid,
                     'time_elapsed': duration}
            llm_stats.append(stats)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    stats_df = pd.DataFrame(llm_stats)
    stats_df.to_csv(args.stats_file, index=False)

if __name__ == "__main__":
    main()