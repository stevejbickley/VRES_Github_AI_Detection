import requests
import json
from datetime import datetime

# Replace with your GitHub Personal Access Token
GITHUB_TOKEN = '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'
REPO_OWNER = 'sindresorhus'
REPO_NAME = 'awesome'

# GitHub API base URL
BASE_URL = 'https://api.github.com'

# Headers for authentication
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_contributors():
    """Fetch the list of contributors for the repository."""
    url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contributors'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch contributors: {response.status_code}")
        return []

def get_commits_for_contributor(username):
    """Fetch all commits for a specific contributor."""
    commits = []
    page = 1
    while True:
        url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/commits'
        params = {
            'author': username,
            'page': page,
            'per_page': 100  # Max number of commits per page
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            page_commits = response.json()
            if not page_commits:
                break
            commits.extend(page_commits)
            page += 1
        else:
            print(f"Failed to fetch commits for {username}: {response.status_code}")
            break
    return commits

def save_commits_to_file(commits, filename):
    """Save the commits to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(commits, f, indent=4)

def main():
    contributors = get_contributors()
    if not contributors:
        print("No contributors found.")
        return

    for contributor in contributors:
        username = contributor['login']
        print(f"Fetching commits for contributor: {username}")
        commits = get_commits_for_contributor(username)
        if commits:
            filename = f'{username}_commits.json'
            save_commits_to_file(commits, filename)
            print(f"Saved {len(commits)} commits to {filename}")
        else:
            print(f"No commits found for {username}")

if __name__ == '__main__':
    main()
