import os
import urllib.request

MODEL_DIR = "models"
DATA_DIR = "wiki_data"
MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/"
    "resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, MODEL_FILENAME)

    if os.path.exists(dest):
        print(f"Model already exists at {dest}")
        return

    print(f"Downloading {MODEL_FILENAME} (~4.9 GB)...")
    urllib.request.urlretrieve(MODEL_URL, dest)
    print("Download complete.")

if __name__ == "__main__":
    main()
