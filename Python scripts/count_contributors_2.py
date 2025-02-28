import requests

# Replace with your actual token
GITHUB_TOKEN = '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'
REPO_OWNER = 'Significant-Gravitas'
REPO_NAME = 'AutoGPT'

BASE_URL = 'https://api.github.com'
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_contributors_count():
    """Fetch the total number of contributors for the repository."""
    url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contributors'
    all_contributors = []
    page = 1
    
    while True:
        response = requests.get(f'{url}?page={page}&per_page=100', headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch contributors: {response.status_code}")
            return 0
            
        page_contributors = response.json()
        if not page_contributors:  # No more contributors to fetch
            break
            
        all_contributors.extend(page_contributors)
        page += 1
    
    return len(all_contributors)

if __name__ == '__main__':
    count = get_contributors_count()
    print(f"Number of contributors for {REPO_OWNER}/{REPO_NAME}: {count}")
