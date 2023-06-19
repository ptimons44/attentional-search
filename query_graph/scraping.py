import config
import requests
from bs4 import BeautifulSoup
import json

def search_google(keywords, before=None, extra_params=None):
    """
    before is string in YYYY-MM-DD format
    """
    if before is not None:
        keywords += f" before:{before}"
    params = {
        "key": config.GGLSEARCH_APIKEY(),
        "cx": config.GGL_SE(),
        "q": keywords,
        "h1": "en",
        "lr": "lang_en"
    }
    if extra_params is not None:
        params |= extra_params

    response = requests.get(config.GGLSEARCH_URL(), params=params)
    try:
        response = json.loads(response.content)
        response["error"] = 0
        
    except:
        response = json.loads("{\"error\": 1}")
    return response["items"]

def get_webpage_content(url):
    try:
        # Send a GET request to the specified URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the webpage
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the textual content from the parsed HTML
            # For example, if you want to get the text from all paragraphs:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])

            return content
        else:
            print("Request failed with status code:", response.status_code)
    except requests.RequestException as e:
        print("An error occurred:", str(e))

    return None

def scrape_urls(urls):
    info = dict()
    for url in urls:
        content = get_webpage_content(url)
        info[url] = content
    return info

if __name__ == "__main__":
    search = search_google("climate change")
    info = scrape_urls(list(item["link"] for item in search))
    print(search)