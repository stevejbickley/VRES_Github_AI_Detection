import requests
import json
import time

# GitHub API token for authentication (replace with your token)
token = "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"

# List of repositories
repositories = [
    {"owner": "freeCodeCamp", "repo": "freeCodeCamp"},
    {"owner": "tensorflow", "repo": "tensorflow"},
    {"owner": "ohmyzsh", "repo": "ohmyzsh"},
    {"owner": "Significant-Gravitas", "repo": "AutoGPT"},
    {"owner": "flutter", "repo": "flutter"},
    {"owner": "EbookFoundation", "repo": "free-programming-books"},
    {"owner": "sindresorhus", "repo": "awesome"},
    {"owner": "codecrafters-io", "repo": "build-your-own-x"},
    {"owner": "public-apis", "repo": "public-apis"},
    {"owner": "jwasham", "repo": "coding-interview-university"},
]

# Function to get commit history
def get_commit_history(owner, repo):
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Authorization": f"token {token}"}
    commits = []

    while commits_url:
        response = requests.get(commits_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch commits for {owner}/{repo}")
            break
        data = response.json()
        commits.extend(data)
        
        # Check for pagination
        if 'next' in response.links:
            commits_url = response.links['next']['url']
        else:
            commits_url = None

    return commits

# Function to get commit details with retries
def get_commit_details(owner, repo, sha, retries=5, backoff_factor=1.0):
    commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {token}"}
    for attempt in range(retries):
        try:
            response = requests.get(commit_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                print(f"Failed to fetch commit details for {owner}/{repo} with SHA: {sha}")
                return None

# Main script
all_commits = []

for repo in repositories:
    owner = repo["owner"]
    repo_name = repo["repo"]
    commits = get_commit_history(owner, repo_name)
    for commit in commits:
        sha = commit["sha"]
        commit_details = get_commit_details(owner, repo_name, sha)
        if commit_details:
            all_commits.append(commit_details)

# Save to JSON file
with open("commits.json", "w") as f:
    json.dump(all_commits, f, indent=4)

print("Commit history saved to commits.json")
