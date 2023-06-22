import config
import requests
from bs4 import BeautifulSoup
import json

def search_google(keywords, num_results=10, timeperiod=None):
    """
    before is string in YYYY-MM-DD format
    """
    output = []
    while num_results > 0:
        if num_results < 10:
            page=1
            num=num_results
        else:
            num=10
            page=1
        params = {
            "key": config.GGLSEARCH_APIKEY(),
            "cx": config.GGL_SE(),
            "q": keywords,
            "h1": "en",
            "lr": "lang_en",
            "page": page,
            "num": num
        }

        response = requests.get(config.GGLSEARCH_URL(), params=params)
        try:
            response = json.loads(response.content)
            response["error"] = 0
            
        except:
            response = json.loads("{\"error\": 1}")
        
        for item in response["items"]:
            output.append(item)
        num_results -= num
        page += 1

    return output

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

def scrape_urls(urls, aggregate_content, query):
    info = dict()
    for url in urls:
        if url in info  or url in aggregate_content:
            continue
        content = get_webpage_content(url)
        if content is not None:
            info[url] = (query, content)
    return info

def get_top_k_content(query, aggregate_content, k=10):
    search = search_google(query, num_results=k)
    return scrape_urls(list(item["link"] for item in search), aggregate_content, query)

if __name__ == "__main__":
    search = search_google("climate change", num_results=3)
    print(len(search))
    info = scrape_urls(list(item["link"] for item in search))
    print(info)
