import os
import json
import requests
import shutil
import subprocess
import javalang
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# GitHub API token (optional, for higher rate limits)
GITHUB_TOKEN = "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"  # Replace with your GitHub token

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def search_github_repos(query, num_repos=5):
    """Search for Java repositories on GitHub based on a query."""
    url = f"https://api.github.com/search/repositories?q={query}+language:java&sort=stars&order=desc"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print("Error fetching repos:", response.json())
        return []
    
    repos = response.json().get("items", [])[:num_repos]
    return [repo["clone_url"] for repo in repos]

def clone_and_extract_java(repo_url, download_path):
    """Clone a GitHub repository and extract all Java files."""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(download_path, repo_name)
    
    if os.path.exists(repo_path):
        print(f"Repository {repo_name} already exists, skipping clone.")
    else:
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    
    java_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
    
    return java_files

def extract_java_methods(java_files):
    """Extract methods from Java files."""
    dataset = []
    
    for file_path in tqdm(java_files, desc="Processing Java files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                java_code = f.read()
            
            tree = javalang.parse.parse(java_code)
            for path, node in tree:
                if isinstance(node, javalang.tree.MethodDeclaration):
                    method_code = java_code[node.position.line - 1 : node.position.line + 10]  # Approx range
                    dataset.append({
                        "method_name": node.name,
                        "parameters": [p.type.name for p in node.parameters],
                        "return_type": node.return_type.name if node.return_type else "void",
                        "code": method_code
                    })
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return dataset

def load_comment_model():
    """Load the Java method comment generation model."""
    model_name = "Salesforce/codet5-small"  # Change to your trained model if needed
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_method_comments(dataset, tokenizer, model):
    """Generate comments for extracted Java methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for entry in tqdm(dataset, desc="Generating comments"):
        method_code = entry["code"]
        input_text = f"Generate a comment: {method_code}"
        
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_length=100)
        
        generated_comment = tokenizer.decode(outputs[0], skip_special_tokens=True)
        entry["generated_comment"] = generated_comment
    
    return dataset

if __name__ == "__main__":
    QUERY = "project"
    NUM_REPOS = 3
    DOWNLOAD_PATH = "java_repos"
    OUTPUT_FILE = "java_comment_dataset.json"

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    
    print("🔍 Searching for Java repositories...")
    repo_urls = search_github_repos(QUERY, NUM_REPOS)
    
    java_files = []
    for repo_url in repo_urls:
        print(f"📥 Cloning {repo_url} ...")
        java_files.extend(clone_and_extract_java(repo_url, DOWNLOAD_PATH))
    
    print(f"🔎 Extracting methods from {len(java_files)} Java files...")
    dataset = extract_java_methods(java_files)
    
    print("🤖 Loading comment generation model...")
    tokenizer, model = load_comment_model()
    
    print("📝 Generating method comments...")
    dataset = generate_method_comments(dataset, tokenizer, model)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"✅ Dataset saved to {OUTPUT_FILE}")
