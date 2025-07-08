#This is used for testing when LLaMA3 via Ollama is not debugged. Also, this is used when the prompt is not found (even after ls -ld or cd ~/ .... , etc.)

prompt = "Explain epilepsy in zebrafish"
import subprocess

def run_llama(prompt):
    proc = subprocess.Popen(
        ["ollama", "run", "llama3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(prompt, timeout=60)
    if err:
        print("[LLAMA ERROR]", err)
    return out.strip()

print(run_llama(prompt))
