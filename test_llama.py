#This script is a test for LLaMA3 for when it doesn't work on the main hybrid_pipeline.py

import subprocess

def analyze_with_llama(prompt):
    process = subprocess.Popen(
        ["ollama", "run", "llama3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    response, error = process.communicate(input=prompt)
    if error:
        print("Error from Ollama:", error)
    return response.strip()

prompt = "Hello LLaMA, say hi!"
print("Sending prompt to LLaMA...")
response = analyze_with_llama(prompt)
print("Response:", response)


