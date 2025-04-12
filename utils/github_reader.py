import re
import requests
import logging

logger = logging.getLogger(__name__)

def extract_github_links_from_text(text):
    pattern = r"https?://(?:www\.)?github\.com/[A-Za-z0-9_-]+"
    return list(set(re.findall(pattern, text)))

def get_github_username_from_url(url):
    match = re.match(r"https?://github\.com/([^/]+)", url)
    return match.group(1) if match else None

def get_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    logger.warning(f"Ошибка получения репозиториев для {username}")
    return []

def get_readme_text(username, repo):
    url = f"https://api.github.com/repos/{username}/{repo}/readme"
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    response = requests.get(url, headers=headers)
    return response.text if response.status_code == 200 else ""

def collect_github_text(github_url):
    username = get_github_username_from_url(github_url)
    if not username:
        return ""
    repos = get_repos(username)
    return " ".join(get_readme_text(username, repo['name']) for repo in repos)