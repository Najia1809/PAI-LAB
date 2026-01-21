import re
import requests
from bs4 import BeautifulSoup

def extract_emails(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        # Regular expression for emails
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        emails = set(re.findall(email_pattern, text))

        return emails

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
