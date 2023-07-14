# TODO: remove when ready to deploy
# import sys
# sys.path.insert(0, "/Users/patricktimons/Documents/GitHub/query-graph")

import re
import string

import requests
from bs4 import BeautifulSoup
import json

# google api
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from threading import Lock
import json
lock = Lock()

import spacy 
from scipy.spatial.distance import cosine

from sentence_transformers import SentenceTransformer, util

## Boiler plate
from transformers import BertTokenizer, BertModel

import numpy as np

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

# using absolute imports for custom modules
from query_graph.gpt import callGPT
from query_graph import config
from query_graph.logger import logger



    
class Parser:
    def __init__(self, researcher, model, tokenizer):
        self.researcher = researcher
        # TODO: uncomment when debugging is complete
        self.search_queries = self.get_k_search_queries(
            self.researcher.query_sentences, 
            self.researcher.gpt_sentences,
            model,
            tokenizer,
            self.researcher.num_search_queries,
            self.researcher.search_query_attention_threshold
        )

    def get_token_idx_to_word(self, token_ids, tokenizer):
        # dictionary mapping token index to (corresponding word, is_child_root) 
        # parent root is for tracking the chain of updates that we will have to do
        token_idx_to_word = {}
        for idx, token in enumerate(token_ids):
            decoded_token = "".join(tokenizer.decode(token).split())
            if decoded_token.startswith("##"):
                is_dependant_morpheme = True
                decoded_word = token_idx_to_word[idx-1][0] + decoded_token[2:]
                # update chain of dependant morphemes
                i = 1
                while (i == 1 or token_idx_to_word[idx-i+1][1]):
                    token_idx_to_word[idx-i] = (decoded_word, token_idx_to_word[idx-i][1])
                    i += 1
            else:
                is_dependant_morpheme = False
                decoded_word = decoded_token
            token_idx_to_word[idx] = (decoded_word, is_dependant_morpheme)
        
        return {k: token_idx_to_word[k][0] for k in token_idx_to_word}
    
    def get_k_search_queries(self, query_sentences, gpt_sentences, model, tokenizer, k, attention_threshold):
        
        # tokens represented as (query_sentence_idx, gpt_sentence_idx, token_idx)
        attention_to_token = {} # keys
        highly_attended_from_token = {} # values / groups

        token_to_word = {}
        ignore_token = {}

        # ignore stop words, punctuation, special tokens, and lone numerals
        ignore = set(stopwords.words('english')) | set(string.punctuation) | {'[CLS]', '[SEP]'} | set("0123456789") 

        for (q_idx, query_sentence) in enumerate(query_sentences):
            for (gpt_idx, gpt_sentence) in enumerate(gpt_sentences):
                inputs = tokenizer.encode_plus(query_sentence, gpt_sentence, return_tensors='pt')

                # Get token ids for decoding back to words later
                token_ids = inputs['input_ids'].numpy().squeeze()
                
                # dictionary mapping token index to (corresponding word, is_child_root) 
                # parent root is for tracking the chain of updates that we will have to do
                token_idx_to_word = self.get_token_idx_to_word(token_ids, tokenizer)
                token_to_word |= {(q_idx, gpt_idx, token_idx): token_idx_to_word[token_idx] for token_idx in token_idx_to_word}

                outputs = model(**inputs)
                last_layer_attentions = outputs.attentions[-1]  # get the last layer's attentions
                avg_attention = last_layer_attentions.squeeze(0).mean(dim=0)  # average attention across heads
                attention_matrix = avg_attention.detach().numpy()

                keep_token = np.array([bool(tokenizer.decode([token_ids[idx]]) not in ignore) for idx in range(attention_matrix.shape[0])])

                attention_to_j = attention_matrix.sum(axis=0, keepdims=False)
                for j in range(len(attention_to_j)):
                    if keep_token[j]:
                        attention_to_token[(q_idx, gpt_idx, j)] = attention_to_j[j]
                    ignore_token[(q_idx, gpt_idx, j)] = not keep_token[j]
                
                mean = attention_matrix.mean(axis=1, keepdims=False, where=keep_token)
                std = attention_matrix.std(axis=1, keepdims=False, where=keep_token)
                highly_attended_from_token |= {
                    (q_idx, gpt_idx, i): {
                        token_to_word[(q_idx, gpt_idx, j)]
                        for j in range(attention_matrix.shape[1])
                        if attention_matrix[i, j] > mean[i]+std[i]*attention_threshold
                        and keep_token[j]
                    }
                    for i in range(attention_matrix.shape[0])
                    if keep_token[i]
                }

        def attention_to_token_key(token):
            if ignore_token[token]:
                return 0
            else:
                return attention_to_token[token]
        top_k_attendees = sorted(attention_to_token, key=attention_to_token_key, reverse=True)[:k]
        
        return [
            " AND ".join(word for word in highly_attended_from_token[attender])
            for attender in top_k_attendees
        ]


class Researcher(object):
    def __init__(self, query, **kwargs):
        self.query = query
        self.threshold = kwargs.get("threshold", 0.4)
        self.similarity_threshold = kwargs.get("similarity_threshold", 12)
        self.results_per_search = kwargs.get("results_per_search", 5)
        self.num_nodes = kwargs.get("num_nodes", 100)
        self.context_window = kwargs.get("context_window", 2)
        self.search_resolution = kwargs.get("search_resolution", 10000) 
        self.num_search_queries = kwargs.get("num_search_queries", 25) 
        self.search_query_attention_threshold = kwargs.get("search_query_attention_threshold", 0) 

        model_name = 'bert-base-uncased'
        model = kwargs.get("bert_model", BertModel.from_pretrained(model_name, output_attentions=True))
        tokenizer = kwargs.get("bert_tokenizer", BertTokenizer.from_pretrained(model_name))
        
        # TODO uncomment
        self.gpt_response = """ChatGPT: To determine the deadliest animals in Australia, I would consider various factors such as the number of human fatalities caused by different species, the toxicity or venomous nature of the animals, and the likelihood or frequency of encounters with these dangerous creatures. It is important to note that a species being deadly does not necessarily mean it is aggressive or inclined to attack humans, but rather that it poses a potential threat due to its natural characteristics.

One of the most feared and deadliest animals in Australia is the saltwater crocodile (Crocodylus porosus). These massive reptiles are known to be highly aggressive and can be found in coastal areas, rivers, and even some open sea areas in the northern parts of Australia. Saltwater crocodiles are responsible for the highest number of reported fatal attacks on humans in the country. They are particularly dangerous as they are excellent swimmers and ambush predators, capable of striking suddenly with their powerful jaws.

Another dangerous animal in Australia is the box jellyfish (Chironex fleckeri). This marine creature, found in the coastal waters of Northern Australia, possesses extremely potent venom in its tentacles. Box jellyfish stings can cause cardiac arrest and death within minutes, making them one of the deadliest creatures in the ocean. While encounters with box jellyfish are rare and there are protective measures in place at popular swimming locations, their presence highlights the need for caution during marine activities.

Australia is also home to a variety of venomous snakes, including the inland taipan (Oxyuranus microlepidotus) and the eastern brown snake (Pseudonaja textilis). The inland taipan is considered the most venomous snake in the world, with its venom being highly potent and capable of causing rapid paralysis and death. Eastern brown snakes, on the other hand, are responsible for the highest number of snakebite-related deaths in Australia. These snakes are commonly found in populated areas, and their bites can lead to cardiovascular collapse and nervous system failure if not treated promptly.

In addition to the above, other notable deadly animals in Australia include the Sydney funnel-web spider (Atrax robustus), known for its highly toxic venom, and the cone snail (Conus species), which are marine mollusks that can deliver venomous stings.

It is crucial to emphasize that while encounters with these deadly animals can and do occur, the likelihood of such encounters is generally quite low. It is important for residents and visitors to Australia to be aware of their surroundings, follow safety protocols, and seek professional assistance in case of any encounters with dangerous wildlife."""

        # self.gpt_response = self.ask_gpt_query(query)
        self.gpt_sentences = Page.split_into_sentences(self, self.gpt_response)
        self.query_sentences = Page.split_into_sentences(self, query)

        parser = Parser(self, model, tokenizer)
        self.search_queries = parser.search_queries
        logger.info(f"Trying the following search queries: {[q for q in self.search_queries]}")


    def ask_gpt_query(self, query):
        with open("query_graph/gpt_prompts.json", "r") as f:
            prompt = json.loads(f.read())["initial prompt"]
        prompt += query
        response = callGPT(prompt)
        return response
    
    def get_k_urls(self, search_query, k, url_to_queries, query_to_urls):
        current_results = 0
        start_index = 1
        while current_results < k:
            try:
                service = build("customsearch", "v1", developerKey=config.GGLSEARCH_APIKEY())
                res = service.cse().list(
                    q=search_query,
                    cx=config.GGL_SE(),
                    hl="en",
                    lr="lang_en",
                    start=start_index,
                    num=min(k-current_results, 10), # Requesting the minimum between remaining results and maximum results per page (10)
                ).execute()
                if 'items' in res:
                    for item in res['items']:
                        url = item['link']
                        with lock: # Adding thread safety
                            if (current_results < k) and (url not in url_to_queries):
                                url_to_queries[url] = {search_query}
                                current_results += 1
                            else:
                                query_set = set(url_to_queries[url])
                                query_set.add(search_query)
                                url_to_queries[url] = query_set

                            if search_query not in query_to_urls:
                                query_to_urls[search_query] = [url]
                            else:
                                # create a regular Python list, update it, and then assign it back to the Manager dictionary
                                url_list = list(query_to_urls[search_query])
                                url_list.append(url)
                                query_to_urls[search_query] = url_list

                start_index += 10 # Update starting index for next results page
            except HttpError as e:
                print(f"An error has occurred: {e}")
                break
            except Exception as e:
                print(f"An unexpected error has occurred: {e}")
                break

    

    def create_page(self, search_queries, url, pages_dict):
        page = Page(search_queries, url)
        pages_dict[url] = page

    def create_pages_and_sentences(self, search_queries, url, sentence_list):
        logger.debug(f"creating page and sentences for {url}")

        # creating page
        page = Page(search_queries, url)
        if page.content:
            for (position, sentence_text) in enumerate(page.sentences):
                sentence_list.append(
                    Sentence(
                        search_queries,
                        sentence_text,
                        page.get_sentence_content(position, self.context_window), # context
                        len(sentence_list), # index
                        url
                    )
                )
                # sentence.embedding = model.encode(sentence.sentence)
                # sentence.relevance = sentence.embedding.dot(self.gpt_response_embedding)
                # sentence_list.append(sentence)
    

class Page():
    def __init__(self, search_queries, url): #content, url, ranking):
        self.search_queries = search_queries # the search query(ies) that returned this page
        self.url = url

        self.content = self.get_webpage_content()
        if self.content:
            self.sentences = self.split_into_sentences(self.content)
            logger.debug(f"page initialized with {len(self.sentences)} sentences from {self.url}")
            if len(self.sentences) > 500:
                logger.debug(f"{url} has more than 800 sentences. Truncating to 800 sentences")
        else:
            self.sentneces = []
        

    def get_webpage_content(self):
        logger.debug(f"getting content from {self.url}")
        try:
            # Send a GET request to the specified URL
            response = requests.get(self.url)
            logger.debug(f"retrieving content from {self.url}")

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the HTML content of the webpage
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract the textual content from the parsed HTML
                # For example, if you want to get the text from all paragraphs:
                paragraphs = soup.find_all('p')
                if len(paragraphs) > 200:
                    logger.debug(f"{self.url} has more than 200 paragraphs. Truncating to 200 paragraphs")
                    content = ' '.join([p.get_text() for p in paragraphs[:200]])
                else:
                    content = ' '.join([p.get_text() for p in paragraphs])
                
                logger.debug(f"returning content from {self.url}")
                return content
            else:
                logger.debug(f"Request failed with status code: {response.status_code}")
        except requests.RequestException as e:
            logger.debug(f"An error occurred: {e}")
        
        logger.debug(f"unable to retrieve content from {self.url}")
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
        # -*- coding: utf-8 -*-
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov|edu|me)"
        digits = "([0-9])"
        multiple_dots = r'\.{2,}'
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
        text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
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
    
    def get_sentence_content(self, position, context_window):
        text = self.sentences[position]
        pre_context, post_context = "", ""
        if position > 0 and context_window > 0:
                pre_context = " ".join(self.sentences[max(0, position-context_window):position]).strip()
        if position < len(self.sentences) - 1 and context_window > 0:
            post_context = " ".join(self.sentences[position+1:min(len(self.sentences)-1, position+context_window+1)]).strip()
        return pre_context + " " + text + " " + post_context
    
class Sentence(Page):
    def __init__(self, search_queries, sentence, context, index, url):
        self.search_queries = search_queries
        self.text = sentence
        self.context = context
        self.index = index
        self.url = url

        # self.similarities = {} # populated in get_top_k_similar_sentences
        # self.relation_to_gpt = {} # populated in get_relation_to_gpt


if __name__ == "__main__":

    query = "What are the deadliest animals in Australia?"
    gpt_response = """ChatGPT: To determine the deadliest animals in Australia, I would consider various factors such as the number of human fatalities caused by different species, the toxicity or venomous nature of the animals, and the likelihood or frequency of encounters with these dangerous creatures. It is important to note that a species being deadly does not necessarily mean it is aggressive or inclined to attack humans, but rather that it poses a potential threat due to its natural characteristics.

One of the most feared and deadliest animals in Australia is the saltwater crocodile (Crocodylus porosus). These massive reptiles are known to be highly aggressive and can be found in coastal areas, rivers, and even some open sea areas in the northern parts of Australia. Saltwater crocodiles are responsible for the highest number of reported fatal attacks on humans in the country. They are particularly dangerous as they are excellent swimmers and ambush predators, capable of striking suddenly with their powerful jaws.

Another dangerous animal in Australia is the box jellyfish (Chironex fleckeri). This marine creature, found in the coastal waters of Northern Australia, possesses extremely potent venom in its tentacles. Box jellyfish stings can cause cardiac arrest and death within minutes, making them one of the deadliest creatures in the ocean. While encounters with box jellyfish are rare and there are protective measures in place at popular swimming locations, their presence highlights the need for caution during marine activities.

Australia is also home to a variety of venomous snakes, including the inland taipan (Oxyuranus microlepidotus) and the eastern brown snake (Pseudonaja textilis). The inland taipan is considered the most venomous snake in the world, with its venom being highly potent and capable of causing rapid paralysis and death. Eastern brown snakes, on the other hand, are responsible for the highest number of snakebite-related deaths in Australia. These snakes are commonly found in populated areas, and their bites can lead to cardiovascular collapse and nervous system failure if not treated promptly.

In addition to the above, other notable deadly animals in Australia include the Sydney funnel-web spider (Atrax robustus), known for its highly toxic venom, and the cone snail (Conus species), which are marine mollusks that can deliver venomous stings.

It is crucial to emphasize that while encounters with these deadly animals can and do occur, the likelihood of such encounters is generally quite low. It is important for residents and visitors to Australia to be aware of their surroundings, follow safety protocols, and seek professional assistance in case of any encounters with dangerous wildlife."""
    # researcher = Researcher(query)
    # researcher.gpt_response = researcher.ask_gpt_query(query)
    # researcher.gpt_sentences = Page.split_into_sentences(researcher, researcher.gpt_response)
    # researcher.query_sentences = Page.split_into_sentences(researcher, query)

    # parser = Parser(researcher)
    # researcher.search_queries = parser.search_queries
