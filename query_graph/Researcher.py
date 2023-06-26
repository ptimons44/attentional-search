import re

import requests
from bs4 import BeautifulSoup
import json

import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

import config

class Researcher:
    def __init__(self, queries, num_results=10):
        self.queries = queries
        self.num_results=num_results
        self.urls = set()
        self.fetch_urls()
        
    def fetch_urls(self):
        for search_query in self.queries:
            search = Search(search_query, self.num_results)
            for url in search.search_google():
                self.urls.add((search_query, url))


class Search(Researcher):
    def __init__(self, search_query, num_results):
        self.search_query = search_query    
        self.num_results = 10

    def search_google(self):
        """
        before is string in YYYY-MM-DD format
        """
        output = []
        while self.num_results > 0:
            if self.num_results < 10:
                page=1
                num=self.num_results
            else:
                num=10
                page=1
            params = {
                "key": config.GGLSEARCH_APIKEY(),
                "cx": config.GGL_SE(),
                "q": self.search_query,
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
            self.num_results -= num
            page += 1

        return list(item["link"] for item in output)


class Page(Researcher):
    # -*- coding: utf-8 -*-
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    def __init__(self, search_query, url): #content, url, ranking):
        self.search_query = search_query
        self.url = url

        self.content = self.get_webpage_content()
        if self.content:
            self.sentences = self.split_into_sentences(self.content)

    def get_webpage_content(self):
        try:
            # Send a GET request to the specified URL
            response = requests.get(self.url)

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


    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split the text into sentences.

        If the text contains substrings "<prd>" or "<stop>", they would lead 
        to incorrect splitting because they are used as markers for splitting.

        :param text: text to be split into sentences
        :type text: str

        :return: list of sentences
        :rtype: list[str]
        """
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(self.prefixes,"\\1<prd>",text)
        text = re.sub(self.websites,"<prd>\\1",text)
        text = re.sub(self.digits + "[.]" + self.digits,"\\1<prd>\\2",text)
        text = re.sub(self.multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + self.alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(self.acronyms+" "+self.starters,"\\1<stop> \\2",text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+self.suffixes+"[.] "+self.starters," \\1<stop> \\2",text)
        text = re.sub(" "+self.suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + self.alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = [s.strip() for s in sentences]
        if sentences and not sentences[-1]: sentences = sentences[:-1]
        return sentences
    

    
class Sentence(Page):
    def __init__(self, search_query, url, position, context_window):
        super().__init__(search_query, url)
        self.position = position
        self.text = self.sentences[position]

        if position > 0 and context_window > 0:
                pre_context = " ".join(self.sentences[max(0, position-context_window):position]).strip()
        if position < len(self.sentences) - 1 and context_window > 0:
            post_context = " ".join(self.sentences[position+1:min(len(self.sentences)-1, position+context_window+1)]).strip()
        self.context = pre_context + " " + self.text + " " + post_context


    def vector(self, context=""):
        # Add the special tokens.
        marked_text = "[CLS] " + self.text + " [SEP]"
        
        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the 22 tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
        with torch.no_grad():

            outputs = self.model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]

        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding

    def similarity(self, query):
        """return a scalar on [0,1] to indicate similarity

        Args:
            query (vector): embedding of search query

        Returns:
            _type_: _description_
        """
        return cosine(self.vector, query)
    
if __name__ == "__main__":
    # queries = ["climate change AND global warming", "climate change OR global warming"]
    queries = ["Mona Lisa AND Leonardo da Vinci", "Mona Lisa or Leonardo da Vinci", "Mathematics AND Leonardo da Vinci", "Mathematics OR Leonardo da Vinci"]
    
    researcher = Researcher(queries)

    import threading

    def create_page(search_query, url, pages):
        page = Page(search_query,url)
        pages.append(page)

    pages = []
    threads = []
    for (search_query, url) in researcher.urls:
        threads.append(threading.Thread(target=create_page, args=(search_query,url,pages)))
    
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    print(len(pages))
    for page in pages:
        if page.content:
            print(page.sentences)
            break

