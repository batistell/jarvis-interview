import os
import subprocess
from git import Repo

REPO_URL = "https://github.com/batistell/face-registry.git"
CLONE_DIR = "face-registry"

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def ensure_repo_cloned():
    project_root = get_project_root()
    repo_path = os.path.join(project_root, CLONE_DIR)
    
    if not os.path.exists(repo_path):
        print(f"[*] Cloned repository 'face-registry' not found. Cloning from {REPO_URL}...")
        try:
            Repo.clone_from(REPO_URL, repo_path)
            print("[+] Repository cloned successfully!")
        except Exception as e:
            print(f"[-] Failed to clone repository: {e}")
            print("[-] Please ensure you are online or clone it manually to: " + repo_path)
    return repo_path

def read_file_safely(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        # If encoding error, try latin-1
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return f"[Error reading file: {e}]"

def index_codebase(include_code=False):
    repo_path = ensure_repo_cloned()
    if not os.path.exists(repo_path):
        return "No codebase context available (face-registry repository missing)."

    indexed_content = []

    # 1. Index root markdown files
    root_mds = ["README.md", "challenge.md", "abc.md"]
    for md in root_mds:
        file_path = os.path.join(repo_path, md)
        if os.path.exists(file_path):
            content = read_file_safely(file_path)
            indexed_content.append(f"=== FILE: {md} ===\n{content}\n")

    # 2. Index docs/ directory
    docs_dir = os.path.join(repo_path, "docs")
    if os.path.exists(docs_dir):
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    content = read_file_safely(file_path)
                    indexed_content.append(f"=== FILE: {rel_path} ===\n{content}\n")

    # 3. Index backend source code (Java files) if requested
    if include_code:
        java_dir = os.path.join(repo_path, "backend", "src", "main", "java")
        if os.path.exists(java_dir):
            for root, _, files in os.walk(java_dir):
                for file in files:
                    if file.endswith(".java"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, repo_path)
                        content = read_file_safely(file_path)
                        indexed_content.append(f"=== FILE: {rel_path} ===\n{content}\n")

    return "\n".join(indexed_content)

def load_style_template():
    project_root = get_project_root()
    style_path = os.path.join(project_root, "interview-example.md")
    
    if os.path.exists(style_path):
        return read_file_safely(style_path)
    else:
        return "No style template found in interview-example.md. Adopt a direct technical tone in Portuguese."

if __name__ == "__main__":
    print("[*] Testing codebase indexer...")
    repo_path = ensure_repo_cloned()
    if os.path.exists(repo_path):
        print(f"[+] Repository path verified: {repo_path}")
        context = index_codebase()
        style = load_style_template()
        print(f"[+] Indexed {len(context)} characters of codebase context.")
        print(f"[+] Loaded {len(style)} characters of style template.")
    else:
        print("[-] Test failed: repository not available.")
