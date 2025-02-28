import requests
import json
import time
import os  # For file operations

# GitHub API token for authentication
token = "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"

# Repository details
owner = "EbookFoundation"
repo = "free-programming-books"

# Function to check rate limit
def check_rate_limit():
    rate_limit_url = "https://api.github.com/rate_limit"
    headers = {"Authorization": f"token {token}"}
    try:
        response = requests.get(rate_limit_url, headers=headers)
        if response.status_code == 200:
            rate_limit_data = response.json()
            remaining = rate_limit_data["resources"]["core"]["remaining"]
            print(f"Remaining requests: {remaining}")
            return remaining
        else:
            print(f"Failed to check rate limit. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return 0
    except Exception as e:
        print(f"Error checking rate limit: {e}")
        return 0

# Function to get commit history
def get_commit_history(owner, repo):
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Authorization": f"token {token}"}
    commits = []
    while commits_url:
        if check_rate_limit() < 10:  # Add a delay if remaining requests are low
            print("Waiting for rate limit reset...")
            time.sleep(60)  # Wait for 1 minute
        response = requests.get(commits_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch commits for {owner}/{repo}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            break
        data = response.json()
        commits.extend(data)
        # Check for pagination
        if 'next' in response.links:
            commits_url = response.links['next']['url']
        else:
            commits_url = None
    # Save to JSON file
    commit_history_file = f"commit_history_{owner}_{repo}.json"
    with open(commit_history_file, "w") as f:
        json.dump(commits, f, indent=4)
    print(f"Commit history saved to {commit_history_file}.")
    return commits

# Function to get commit details
def get_commit_details(owner, repo, sha):
    commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {token}"}
    try:
        response = requests.get(commit_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch commit details for SHA: {sha}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching commit details: {e}")
        return None

# Main script
def main():
    # Fetch commit history
    commit_history_file = f"commit_history_{owner}_{repo}.json"
    if os.path.exists(commit_history_file):
        print(f"Commit history file already exists: {commit_history_file}. Loading from file...")
        with open(commit_history_file, "r") as f:
            commits = json.load(f)
    else:
        print(f"Fetching commit history for {owner}/{repo}...")
        commits = get_commit_history(owner, repo)

    # Fetch commit details
    all_commit_details = []
    for idx, commit in enumerate(commits, 1):
        if "sha" not in commit:
            print(f"Skipping commit {idx}: Missing 'sha' key.")
            continue
        sha = commit["sha"]
        print(f"Processing commit {idx}/{len(commits)}: SHA {sha}")
        commit_details = get_commit_details(owner, repo, sha)
        if commit_details:
            all_commit_details.append(commit_details)
        time.sleep(1)  # Delay to avoid rate limiting

    # Save commit details
    commit_details_file = f"commit_details_{owner}_{repo}.json"
    with open(commit_details_file, "w") as f:
        json.dump(all_commit_details, f, indent=4)
    print(f"Saved {len(all_commit_details)} commit details to {commit_details_file}.")

if __name__ == "__main__":
    main()
