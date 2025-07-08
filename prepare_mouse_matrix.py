#This was a major testing session for the first GSE264537. Using this script was able to return a full row and columns of genes and IDs which was the usage of a matrix. This script used the .csv file to eventually create a matrix and store them into the results.

import pandas as pd
import os

INPUT = "data/raw/GSE264537_raw_counts.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT)

# Extract gene names
gene_names = df["Gene.name"].values
sample_columns = df.columns[1:-1]  # skip 'ID' and 'Gene.name'

# Create expression matrix: samples as rows, genes as columns
expression_data = df[sample_columns].T
expression_data.columns = gene_names
expression_data.index.name = "SampleID"
expression_data.reset_index(inplace=True)

# Save expression matrix
expression_data.to_csv(f"{OUTPUT_DIR}/mouse_expression_matrix.csv", index=False)

# Create labels: 0 = WT, 1 = KO
def label_from_sample(sample_id):
    return 1 if ".KO." in sample_id else 0

labels = pd.DataFrame({
    "SampleID": sample_columns,
    "Label": [label_from_sample(s) for s in sample_columns]
})
labels.to_csv(f"{OUTPUT_DIR}/mouse_labels.csv", index=False)

print("[✔] Expression matrix and labels saved to data/processed/")

