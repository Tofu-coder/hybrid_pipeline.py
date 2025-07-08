# hybrid_pipeline.py
This is the main pipeline that will be used for the AI-Neurological-Disorder-Project. This integrates two ML and AI programs which are LLaMA3 via Ollama and BioGPT via OpenAI. There are multiple scripts that correspond to the main hybrid_pipeline.py (which is the Local LLM) and is parallel-processed for HPC scaling.

IMPORTS:

****import os, subprocess, time
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM**

- os: For file and directory operations.
- subprocess: To run external shell commands (like calling Ollama LLaMA).
- time: (Imported but not actively used in current script — could be for future timing).
- torch: PyTorch framework used for BioGPT inference.
- tensorflow: For running a simple TensorFlow ML model.
- transformers: Hugging Face library to load BioGPT tokenizer and model.

Configuration and Directories:

**DATA_DIR = "data/raw"
PROMPT_DIR = "prompts"
RESULTS_DIR = "results"
SKIP_DOWNLOAD = True  # Skip Entrez + GEO downloading

BIOMODEL = "microsoft/BioGPT-Large"
OLLAMA_MODEL = "llama3"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)**

- Defines locations for your raw data, prompt templates, and output results.
- SKIP_DOWNLOAD controls whether to fetch GEO datasets (currently set to skip).
- Model names for BioGPT and Ollama LLaMA are assigned here.
- Creates the directories if they don’t exist to avoid errors later.

Load BioGPT Model and Tokenizer:

**print("[INFO] Loading BioGPT model...")
tokenizer = AutoTokenizer.from_pretrained(BIOMODEL)
biogpt = AutoModelForCausalLM.from_pretrained(BIOMODEL)
**

- Loads the pretrained BioGPT tokenizer and model for biomedical causal language modeling.
- This enables you to generate biomedical text responses given an input.

Define a Simple TensorFlow Model:

**TF_MODEL = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
TF_MODEL.compile(optimizer='adam', loss='binary_crossentropy')**

Defines a tiny neural network with:
- Input: 3 features (dummy vector size).
- One hidden dense layer with 4 neurons, ReLU activation.
- Output layer with sigmoid activation for binary score.
- Compiles the model with Adam optimizer and binary cross-entropy loss.
- This is a placeholder simple ML model to demonstrate integration with TensorFlow.


run_llama()- Run LLaMA via Ollama CLI:

def run_llama(prompt: str) -> str:
    print("[DEBUG] Starting run_llama()")
    for attempt in range(2):  # retry once
        proc = subprocess.Popen(
            ["ollama", "run", OLLAMA_MODEL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            out, err = proc.communicate(prompt, timeout=90)
            if err.strip():
                print("[LLAMA ERROR]", err.strip())
            return out.strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            print("[ERROR] Ollama call timed out.")
    return ""

- Calls the external ollama run llama3 command.
- Passes the prompt string to Ollama via stdin.
- Waits for output with a 90-second timeout.
- Retries once if timed out.
- Returns LLaMA-generated text or empty string on failure.
- Prints debug/error messages for tracing.

run_biogpt - Run BioGPT Inference with Transformers:

**def run_biogpt(text: str) -> str:
    try:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outs = biogpt.generate(**inputs, max_new_tokens=128)
            return tokenizer.decode(outs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] run_biogpt exception: {e}")
        return ""**

- Tokenizes the input text.
- Runs BioGPT's generate method to produce up to 128 tokens of new text.
- Decodes and returns the generated text.
- Catches and prints any exceptions.

run_tf() - Run TensorFlow Model Inference:

**def run_tf(vec):
    try:
        return float(TF_MODEL(tf.convert_to_tensor([vec], tf.float32)).numpy()[0][0])
    except Exception as e:
        print(f"[ERROR] run_tf exception: {e}")
        return 0.0**

- Takes a vector of 3 floats.
- Converts to TensorFlow tensor.
- Runs the model to get a prediction scalar.
- Returns it as a Python float.
- On error, prints and returns 0.0.

Utility Function clean():

**def clean(name):
    return name.replace(".", "_").replace(" ", "_")**

- Replaces dots and spaces in filenames with underscores for safe output filenames.

Main Pipeline Logic (main() function):

**def main():
    if not SKIP_DOWNLOAD:
        print("[PIPELINE] Downloading GEO datasets")
        from Bio import Entrez
        Entrez.email = "you@example.com"  # Update with your email

        def download_geo(geo_id):
            from Bio import Entrez
            handle = Entrez.esearch(db="gds", term=f"{geo_id}[Accession] AND suppFile[Filter]")
            record = Entrez.read(handle)
            handle.close()
            # You can add download logic back if needed

        GEO_IDS = ["GSE264537", "GSE26246", "GSE43312", "GSE48054", "GSE275235"]
        for gid in GEO_IDS:
            download_geo(gid)**

- If downloading enabled, uses Biopython Entrez to search GEO for supplementary files (currently no actual downloading).
- Lists GEO accession IDs to query.

  **  print("\n[PIPELINE] Starting hybrid pipeline")

    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".tsv", ".csv"))]
    prompts = [f for f in os.listdir(PROMPT_DIR) if f.endswith(".txt")]

    pairs = []
    for df in data_files:
        for pr in prompts:
            if pr.lower().replace("_prompt.txt", "") in df.lower():
                pairs.append((df, pr))

    print(f"[INFO] Total pairs: {len(pairs)}")
    if not pairs:
        print("[WARN] No data-prompt pairs matched.")
        return**

- Lists all data files and prompt templates.
- Pairs them if the prompt filename matches part of the data filename (case-insensitive).
- Warns and stops if no pairs found.

    **for idx, (df, pr) in enumerate(pairs, 1):
        print(f"\n[PROCESSING {idx}/{len(pairs)}] Data file: {df}, Prompt file: {pr}")
        df_path = os.path.join(DATA_DIR, df)
        pr_path = os.path.join(PROMPT_DIR, pr)

        try:
            data = open(df_path).read()[:512]
        except Exception as e:
            print(f"[ERROR] reading data file {df}: {e}")
            continue

        try:
            template = open(pr_path).read()
        except Exception as e:
            print(f"[ERROR] reading prompt {pr}: {e}")
            continue

        if "{data}" not in template:
            print(f"[WARN] Template missing {{data}} placeholder: {pr}")
            continue

        prompt = template.replace("{data}", data)
        llama_output = run_llama(prompt)
        print(f"[LLaMA OUTPUT]: {llama_output[:60]}...")

        biogpt_output = run_biogpt(llama_output)
        print(f"[BioGPT OUTPUT]: {biogpt_output[:60]}...")

        score = run_tf([0.2, 0.4, 0.6])

        result = (
            f"=== {pr} x {df} ===\n\n"
            f"[LLaMA]\n{llama_output}\n\n"
            f"[BioGPT]\n{biogpt_output}\n\n"
            f"[TF SCORE] {score:.4f}"
        )
        out_file = os.path.join(RESULTS_DIR, f"{clean(df)}__{clean(pr)}_hybrid.txt")
        with open(out_file, "w") as f:
            f.write(result)
        print("[DONE]", out_file)

    print("\n[PIPELINE] Completed.")

For each matched pair:
- Reads first 512 chars from the data file (sample input).
- Reads the prompt template and checks it contains {data} placeholder.
- Inserts the data snippet into the prompt.
- Runs LLaMA via Ollama, prints a preview of output.
- Feeds LLaMA output into BioGPT, prints a preview.
- Runs dummy TensorFlow model on a fixed vector, gets a score.
- Writes combined results to a uniquely named file in results directory.
- Prints a done message per pair.
After all, prints pipeline completion.

Script Entry Point:

**if __name__ == "__main__":
    main()**

- Runs the main() function if script is executed directly.

Conclusion:

This pipeline:

(Kind of)fetches GEO dataset metadata.
Finds pairs of data files and corresponding prompt templates.

For each pair:
- Combines data with a prompt template.
- Generates text with LLaMA.
- Feeds LLaMA output into BioGPT for further biomedical generation.
- Runs a simple TF model to produce a score.
- Saves results for analysis.
- It's designed as a hybrid AI pipeline combining external LLaMA (via Ollama CLI), Hugging Face BioGPT, and a dummy TensorFlow model on biological data prompts.
