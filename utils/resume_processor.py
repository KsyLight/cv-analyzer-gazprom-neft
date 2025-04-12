from utils.cv_reader import read_resume_from_file, preprocess_text
from utils.github_reader import extract_github_links_from_text, collect_github_text

def process_resume_text(file_path):
    text = read_resume_from_file(file_path)
    if not text:
        return None

    resume_clean_text = preprocess_text(text)
    github_links = extract_github_links_from_text(text)

    github_combined = ""
    for link in github_links:
        gh_text = collect_github_text(link)
        if gh_text:
            github_combined += " " + preprocess_text(gh_text)

    return resume_clean_text + " " + github_combined if github_combined else resume_clean_text