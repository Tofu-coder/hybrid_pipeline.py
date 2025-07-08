#This was the local LLM that was created only with LLaMA3 via Ollama. This was only local so it didn't gain any public data from other sources

import os
import sys
import subprocess
from glob import glob

print("[START] agent_llama.py launched")  # Top-level debug message

def analyze_with_llama(prompt):
    print("[INFO] analyze_with_llama called")
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=prompt)

        print(f"[DEBUG] STDOUT:\n{stdout}")
        print(f"[DEBUG] STDERR:\n{stderr}")

        if not stdout.strip():
            print("[WARNING] No response from LLaMA.")
            return None

        return stdout.strip()

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return None

def main():
    print("[START] main() function entered")
    data_dir = "data/raw"
    prompt_dir = "prompts"
    output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)

    data_files = sorted(glob(os.path.join(data_dir, "*.tsv")))
    prompt_files = sorted(glob(os.path.join(prompt_dir, "*.txt")))

    print(f"[INFO] Found {len(data_files)} data files and {len(prompt_files)} prompt files")

    if not data_files or not prompt_files:
        print("[ERROR] Missing input files. Exiting.")
        return

    for data_file in data_files:
        print(f"[INFO] Reading {data_file}")
        with open(data_file, "r") as f:
            data = f.read().strip()

        for prompt_file in prompt_files:
            print(f"[INFO] Using template {prompt_file}")
            with open(prompt_file, "r") as f:
                template = f.read()

            if "{data}" not in template:
                print(f"[WARNING] Placeholder not found in {prompt_file}")
                continue

            prompt = template.replace("{data}", data)
            response = analyze_with_llama(prompt)

            if not response:
                print(f"[ERROR] Empty response for {data_file} + {prompt_file}")
                continue

            data_name = os.path.splitext(os.path.basename(data_file))[0]
            prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
            output_file = os.path.join(output_dir, f"{data_name}__{prompt_name}_analysis.txt")

            with open(output_file, "w") as f:
                f.write(response)

            print(f"[SUCCESS] Output saved to {output_file}")

if __name__ == "__main__":
    print("[DEBUG] Running __main__ section")
    main()

