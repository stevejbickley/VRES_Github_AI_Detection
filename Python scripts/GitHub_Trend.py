import requests

# List of AI-related keywords and libraries
ai_keywords = ["AI", "machine learning", "deep learning", "GPT", "pytorch/pytorch", "tensorflow/tensorflow", "Hugging Face Transformers", "onnxruntime", "sentencepiece", "GPU"]

# Function to check if a repository is AI-related
def is_ai_related(repo):
    url = f"https://api.github.com/repos/{repo['full_name']}/contents/.github/workflows"
    response = requests.get(url)
    if response.status_code == 200:
        workflows = response.json()
        for workflow in workflows:
            file_content = requests.get(workflow['download_url']).text
            if any(keyword in file_content for keyword in ai_keywords):
                return True
    return False

# Fetch trending repositories
response = requests.get("https://api.github.com/search/repositories?q=stars:>1000&sort=stars&order=desc")
trending_repos = response.json()['items']

# Classify repositories
ai_repos = []
non_ai_repos = []

for repo in trending_repos:
    if is_ai_related(repo):
        ai_repos.append(repo)
    else:
        non_ai_repos.append(repo)

# Extract top 5 AI and Non-AI repositories
top_5_ai_repos = ai_repos[:5]
top_5_non_ai_repos = non_ai_repos[:5]

print("Top 5 AI Repositories:")
for repo in top_5_ai_repos:
    print(repo['full_name'], repo['html_url'])

print("\nTop 5 Non-AI Repositories:")
for repo in top_5_non_ai_repos:
    print(repo['full_name'], repo['html_url'])
