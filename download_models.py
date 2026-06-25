import os
import sys
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

def main():
    print("Initializing local offline model download...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_dir, "models")
    
    # 1. Download Whisper base model
    whisper_dir = os.path.join(models_dir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    print(f"[*] Downloading Whisper 'base' model to: {whisper_dir}")
    try:
        WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            download_root=whisper_dir
        )
        print("[+] Whisper model successfully downloaded!")
    except Exception as e:
        print(f"[-] Error downloading Whisper model: {e}", file=sys.stderr)
        
    # 2. Download Qwen 2.5 1.5B Instruct CT2 model
    llm_dir = os.path.join(models_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    print(f"[*] Downloading Qwen2.5 1.5B Instruct CT2 model to: {llm_dir}")
    try:
        snapshot_download(
            repo_id="jncraton/Qwen2.5-1.5B-Instruct-ct2-int8",
            local_dir=llm_dir
        )
        print("[+] Qwen LLM model successfully downloaded!")
    except Exception as e:
        print(f"[-] Error downloading Qwen LLM model: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("[+] All models downloaded and installed in the project folder!")

if __name__ == "__main__":
    main()
