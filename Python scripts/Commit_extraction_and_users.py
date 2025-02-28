import requests
import json
import os
import time


# GitHub API token for authentication 
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
    # Save to JSON file
    with open(f"commit_history_{owner}_{repo}.json", "w") as f:
        json.dump(commits, f, indent=4)
    print(f"Commit history saved to commits_history_{owner}_{repo}.json")
    return commits

# Function to get commit details
def get_commit_details(owner, repo, sha):
    commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(commit_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch commit details for {owner}/{repo} with SHA: {sha}")
        return None
    
# Main script
#all_commits = []

for repo in repositories:
    all_commits = []
    owner = repo["owner"]
    repo_name = repo["repo"]
    if os.path.exists(f"commit_details_{owner}_{repo_name}.json"):
        continue
    if os.path.exists(f"commit_history_{owner}_{repo_name}.json"):
        commits = json.load(f"commit_history_{owner}_{repo_name}.json")
    else:
        commits = get_commit_history(owner, repo_name)
    for commit in commits:
        sha = commit["sha"]
        commit_details = get_commit_details(owner, repo_name, sha)
        if commit_details:
            all_commits.append(commit_details)
        time.sleep(1) # Can increase this for rate-limiting
    with open("commits_details_{owner}_{repo_name}.json", "w") as f:
        json.dump(all_commits, f, indent=4)
    print(f"Commit details saved to commits_details_{owner}_{repo_name}.json")




# For manual testing:
for commit in commits:
    sha = commit["sha"]
    commit_details = get_commit_details(owner, repo_name, sha)
    if commit_details:
        all_commits.append(commit_details)
    time.sleep(1) # Can increase this for rate-limiting

# (OPTIONAL)  Save after manual testing...
with open("commits_details_{owner}_{repo_name}.json", "w") as f:
    json.dump(all_commits, f, indent=4)




# Save to JSON file
#with open("commits.json", "w") as f:
#    json.dump(all_commits, f, indent=4)

print("Commit history saved to commits.json")
