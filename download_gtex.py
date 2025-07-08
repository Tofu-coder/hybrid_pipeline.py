#This script is used to gain the GTEx API key. However, this was originally used to gun zip the .TAR, .txt, .csv, .tsv, etc. URL is in place for when the main pipeline will be in place for HPC scaling.

import urllib.request
import gzip
import shutil
import os

url = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
dest_gz = "data/raw/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
dest_gct = dest_gz[:-3]

os.makedirs("data/raw", exist_ok=True)

print("[INFO] Downloading GTEx TPM GCT...")
urllib.request.urlretrieve(url, dest_gz)

print("[INFO] Unzipping...")
with gzip.open(dest_gz, 'rb') as f_in:
    with open(dest_gct, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"[DONE] Saved GCT to {dest_gct}")

