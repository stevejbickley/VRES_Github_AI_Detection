import requests

# Replace this with your actual token from GitHub
GITHUB_TOKEN = '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'
REPO_OWNER = 'Significant-Gravitas'
REPO_NAME = 'AutoGPT'

BASE_URL = 'https://api.github.com'
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_contributors_count():
    """Fetch the number of contributors for the repository."""
    url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contributors'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contributors = response.json()
        return len(contributors)
    else:
        print(f"Failed to fetch contributors: {response.status_code}")
        return 0

if __name__ == '__main__':
    count = get_contributors_count()
    print(f"Number of contributors for {REPO_OWNER}/{REPO_NAME}: {count}")
