#This scripts automates the retrieval, extraction, and preperation of GEO gene expression datasets and generates prompt templates for AI-based analysis.

import os
import ftplib
import gzip
import shutil
import glob

# === Configuration ===
DATA_DIR = "data/raw"
PROMPT_DIR = "prompts"
GSE_IDS = [
    "GSE264537", "GSE261878", "GSE26246",
    "GSE43312", "GSE48054", "GSE275235"
]

# === Dataset Summaries for Prompt Generation ===
summaries = {
    "GSE264537": "Mouse epilepsy dataset examining gene expression in brain regions over time.",
    "GSE261878": "Mouse NPC (neural progenitor cell) gene expression to study neurological development.",
    "GSE26246": "Drosophila model of DRPLA using Atro gene polyQ expansions to study neurodegeneration.",
    "GSE43312": "Zebrafish lipidomics dataset linked to neurological and metabolic disorders.",
    "GSE48054": "Zebrafish model investigating metabolic disruptions in relation to seizures.",
    "GSE275235": "Zebrafish-based transcriptomics to identify biomarkers and drug targets for epilepsy."
}

# === Ensure directories exist ===
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)

# === Function to download series_matrix files from GEO FTP ===
def fetch_geo_series_matrix(gse_id):
    print(f"[FETCH] {gse_id}")
    ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov")
    ftp.login()
    base_path = f"/geo/series/{gse_id[:-3]}nnn/{gse_id}/matrix/"
    try:
        ftp.cwd(base_path)
        files = ftp.nlst()
        for file in files:
            if "series_matrix" in file and file.endswith(".gz"):
                local_path = os.path.join(DATA_DIR, file)
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {file}", f.write)
                print(f"[DOWNLOADED] {file}")
    except Exception as e:
        print(f"[ERROR] {gse_id} - {e}")
    ftp.quit()

# === Extract all .gz files in data/raw ===
def extract_gz_files():
    for gz_file in glob.glob(f"{DATA_DIR}/*.gz"):
        txt_path = gz_file[:-3]
        with gzip.open(gz_file, 'rb') as f_in, open(txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file)
        print(f"[EXTRACTED] {gz_file} -> {txt_path}")

# === Generate prompt templates with real summaries ===
def create_prompt_templates():
    template_base = (
        "The following data represents gene expression profiles from a neurological disorder model. "
        "Please summarize the key genes, suggest possible pathways affected, and identify any markers relevant to epilepsy or neurodegeneration.\n\n"
        "DATA:\n{data}\n\n"
    )
    for gse, summary in summaries.items():
        prompt_path = os.path.join(PROMPT_DIR, f"{gse}_template.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(f"[DATASET SUMMARY]: {summary}\n\n{template_base}")
        print(f"[CREATED] Prompt for {gse} -> {prompt_path}")

# === Run everything ===
def main():
    print("[SETUP] Starting GEO data and prompt preparation...\n")
    for gse in GSE_IDS:
        fetch_geo_series_matrix(gse)
    print("\n[STEP] Extracting .gz files...")
    extract_gz_files()
    print("\n[STEP] Generating prompt templates...")
    create_prompt_templates()
    print("\n✅ All data and prompts are ready in 'data/raw/' and 'prompts/'")

if __name__ == "__main__":
    main()
# This script downloads GEO datasets, extracts them, and creates prompt templates for further analysis.
# It uses the GEO FTP server to fetch series_matrix files, extracts .gz files, and generates prompt templates
# with summaries of each dataset for use in bioinformatics analysis pipelines.

