import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
import pandas as pd
from tqdm import tqdm

load_dotenv()

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

def main():
    parser = argparse.ArgumentParser(description="Extract botanical data using Gemini.")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Name of the Gemini model to use")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input text files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save JSON results")
    parser.add_argument("--stats_file", type=str, default="gemini_stats.csv", help="Filename for the usage statistics CSV")
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    stats_file = args.stats_file

    output_dir.mkdir(parents=True, exist_ok=True)

    llm_stats = []
    
    if not input_dir.exists():
        print(f"Error: The directory '{input_dir}' was not found.")
        return

    agent = Agent(
        args.model_name,
        output_type=BookPlateData,
        model_settings=ModelSettings(temperature=0.0),
        system_prompt=(
            "You are an expert at extracting botanical specimen data from book pages. "
            "Extract all requested information accurately from the provided text."
            "If a specific piece of information is not found in the text, return an empty string "
            "for that field. Do NOT use 'Unknown', 'N/A', or similar placeholders."
        )
    )

    text_files = list(input_dir.glob("*.txt"))

    if args.test:
        text_files = text_files[:5]
    
    if not text_files:
        print(f"No .txt files found in {input_dir}")
        return

    for file_path in tqdm(text_files, desc="Processing Files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sample_text = f.read()

            if not sample_text.strip():
                continue

            start_time = time.perf_counter()
            
            result = agent.run_sync(sample_text)
            
            end_time = time.perf_counter()
            duration = end_time - start_time

            extracted_data = result.output
            usage = result.usage()

            stats = {
                'pageid': file_path.stem,
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'time_elapsed': duration
            }
            llm_stats.append(stats)

            output_filename = file_path.stem + ".json"
            output_path = output_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_data.model_dump_json(indent=2))
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    stats_df = pd.DataFrame(llm_stats)
    stats_df.to_csv(stats_file, index=False)
    print("All processing complete.")

if __name__ == "__main__":
    main()