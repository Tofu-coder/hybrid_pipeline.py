#This was the first ever LLaMA3 via Ollama run and was used to send a debug for whenever Ollama failed.

import subprocess

# Load prompt template and data
prompt_template = open("test_prompt.txt").read()
input_data = open("test_data.tsv").read()

# Replace placeholder
final_prompt = prompt_template.replace("{data}", input_data)

print("[INFO] Final prompt being sent to Ollama:")
print("=" * 40)
print(final_prompt)
print("=" * 40)

# Run Ollama
process = subprocess.Popen(
    ["ollama", "run", "llama3"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
response, error = process.communicate(input=final_prompt)

# Output logs
print("\n[STDOUT]:")
print(response)

print("\n[STDERR]:")
print(error)

