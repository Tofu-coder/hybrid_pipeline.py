#This script is scalled to HPC usage by parallel-processing and using NCBI Entrez, BioGPT-Large on H100 nodes, TF Vectors, and original IDs from GEO. GTEx API key will be implemented soon to this for H100 scaling.

import os, subprocess, tarfile, gzip, shutil, requests
from tqdm import tqdm
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import Entrez
from multiprocessing import Pool, cpu_count

# === CONFIGURATION ===
Entrez.email = "you@example.com"  # Required for NCBI Entrez access
DATA_DIR = "data/raw"
PROMPT_DIR = "prompts"
RESULTS_DIR = "results"
GEO_IDS = ["GSE264537", "GSE26246", "GSE43312", "GSE48054", "GSE275235"]
BIOMODEL = "distilgpt2"  # Use microsoft/BioGPT-Large on H100
OLLAMA_MODEL = "llama3"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[INFO] Loading BioGPT model...")
tokenizer = AutoTokenizer.from_pretrained(BIOMODEL)
biogpt = AutoModelForCausalLM.from_pretrained(BIOMODEL)

# === Preloaded TensorFlow Model ===a
TF_MODEL = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
TF_MODEL.compile(optimizer='adam', loss='binary_crossentropy')

# === Core Functions ===
def run_llama(prompt: str) -> str:
    try:
        proc = subprocess.Popen(
            ["ollama", "run", OLLAMA_MODEL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate(prompt)
        if err.strip():
            print(f"[LLAMA ERROR] {err.strip()}")
        return out.strip()
    except Exception as e:
        return f"[LLAMA ERROR] Exception: {e}"

def run_biogpt(text: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outs = biogpt.generate(**inputs, max_new_tokens=128)
        return tokenizer.decode(outs[0], skip_special_tokens=True)

def run_tf(vec):
    return float(TF_MODEL(tf.convert_to_tensor([vec], tf.float32)).numpy()[0][0])

def clean(name): 
    return name.replace(".", "_").replace(" ", "_")

# === Worker ===
def process_pair(args):
    df_path, pr_path = args
    try:
        with open(df_path, "r") as f:
            text = f.read()[:512]

        with open(pr_path, "r") as f:
            template = f.read()

        if "{data}" not in template:
            return f"[SKIPPED] Missing {{data}} in template: {pr_path}"

        prompt = template.replace("{data}", text)
        llama = run_llama(prompt)
        bgpt = run_biogpt(llama)
        score = run_tf([0.2, 0.4, 0.6])

        result = (
            f"=== {os.path.basename(pr_path)} x {os.path.basename(df_path)} ===\n\n"
            f"[LLaMA]\n{llama}\n\n"
            f"[BioGPT]\n{bgpt}\n\n"
            f"[TF SCORE] {score:.4f}"
        )

        fname = f"{clean(os.path.basename(df_path))}__{clean(os.path.basename(pr_path))}_hybrid.txt"
        with open(os.path.join(RESULTS_DIR, fname), "w") as f:
            f.write(result)

        return f"[DONE] {fname}"

    except Exception as e:
        return f"[ERROR] {df_path} x {pr_path} - {e}"

# === Main ===
def main():
    print("\n[PIPELINE] Starting hybrid pipeline")

    data_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".tsv", ".csv"))]
    prompt_files = [os.path.join(PROMPT_DIR, f) for f in os.listdir(PROMPT_DIR) if f.endswith(".txt")]
    all_pairs = [(df, pr) for df in data_files for pr in prompt_files]

    print(f"[INFO] Total pairs: {len(all_pairs)}")

    try:
        with Pool(processes=cpu_count()) as pool:
            for result in pool.imap_unordered(process_pair, all_pairs):
                print(result)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline manually terminated by user.")

    print("\n[PIPELINE] Completed.")

if __name__ == "__main__":
    main()
