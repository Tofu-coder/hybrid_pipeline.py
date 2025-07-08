#This is a very important script that at first analyzed properly via LLaMA3 running Ollama, BioGPT, and GEO GSE files. Also, this script returns an output of PCA and t/SNE files (graphs and dot-plots). However, this was only done for one genome: GSE264537

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

df = pd.read_csv("data/raw/GSE264537_raw_counts.csv")

# Drop any non-numeric columns (like 'gene_id', 'description', etc.)
df_numeric = df.select_dtypes(include=["number"])

# Keep only genes with reasonable expression
df_numeric = df_numeric.loc[(df_numeric.sum(axis=1) > 10)]

# PCA + t-SNE
X = df_numeric.values
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30)
tsne_result = tsne.fit_transform(X)

# Save plots
os.makedirs("results/analysis_mouse", exist_ok=True)
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1]).set(title="PCA - Mouse Hippocampus")
plt.savefig("results/analysis_mouse/pca_mouse.png")
plt.clf()

sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1]).set(title="t-SNE - Mouse Hippocampus")
plt.savefig("results/analysis_mouse/tsne_mouse.png")
plt.clf()

print("[DONE] PCA and t-SNE plots saved to: results/analysis_mouse")

