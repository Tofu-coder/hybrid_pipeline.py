#This script was to test the hybrid pipeline and see if the first GSE file would compute and analyze the prompts (saved in /prompts). 

import os
from bioinformatics.agent.agent import run_hybrid_pipeline

if __name__ == "__main__":
    print("[INFO] Starting GSE264537 hybrid pipeline run...")

    input_file = "data/raw/GSE264537_raw_counts.csv"
    prompt_file = "prompts/test_prompt.txt"

    run_hybrid_pipeline(expression_file=input_file, prompt_file=prompt_file)

    print("[INFO] GSE264537 hybrid pipeline run completed.")
