# TODO: remove when ready to deploy
# import sys
# sys.path.insert(0, "/Users/patricktimons/Documents/GitHub/query-graph")

from nltk import sent_tokenize

import requests
from bs4 import BeautifulSoup
import json

# google api
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from threading import Lock
import json
lock = Lock()

from scipy.spatial.distance import cosine

from sentence_transformers import SentenceTransformer, util


import numpy as np


# using absolute imports for custom modules
from query_graph.gpt import callGPT
from query_graph import config
from query_graph.logger import logger

from redis import Redis
redis_client = Redis()


class Researcher(object):
    def __init__(self, query, **kwargs):
        self.query = query
        self.threshold = kwargs.get("threshold", 0.4)
        self.similarity_threshold = kwargs.get("similarity_threshold", 12)
        self.results_per_search = kwargs.get("results_per_search", 5)
        self.n_sents = kwargs.get("n_sents", 100)
        self.context_window = kwargs.get("context_window", 2)
        self.search_resolution = kwargs.get("search_resolution", 10000) 
        self.num_search_queries = kwargs.get("num_search_queries", 25) 
        self.search_query_attention_threshold = kwargs.get("search_query_attention_threshold", 0) 
        
        self.gpt_response = kwargs.get("gpt_response", self.ask_gpt_query(query))
        self.gpt_sentences = kwargs.get("gpt_sentences", Page.split_into_sentences(self, self.gpt_response))
        self.query_sentences = kwargs.get("query_sentences", Page.split_into_sentences(self, query))

        if "search_queries" in kwargs:
            self.search_queries = kwargs.get("search_queries")
        else:
            self.search_queries = self.get_k_search_queries(
                self.query_sentences, 
                self.gpt_sentences, 
                self.num_search_queries, 
                self.search_query_attention_threshold
            )
            logger.info(f"Researcher initialized with the following search queries: {self.search_queries}")

        self.url_to_queries = kwargs.get("url_to_queries", {})
        self.query_to_urls = kwargs.get("query_to_urls", {})
        self.sentences = [Sentence.from_dict(s) for s in kwargs.get("sentences", [])]
        
        

    def to_dict(self):
        researcher_dict = {
            "query": self.query,
            "threshold": self.threshold,
            "similarity_threshold": self.similarity_threshold,
            "results_per_search": self.results_per_search,
            "n_sents": self.n_sents,
            "context_window": self.context_window,
            "search_resolution": self.search_resolution,
            "num_search_queries": self.num_search_queries,
            "search_query_attention_threshold": self.search_query_attention_threshold,
            "gpt_response": self.gpt_response,
            "gpt_sentences": self.gpt_sentences,
            "query_sentences": self.query_sentences,
            "search_queries": self.search_queries,
            "url_to_queries": self.url_to_queries,
            "query_to_urls": self.query_to_urls,
        }
        if hasattr(self, "sentences"):
            researcher_dict["sentences"] = [sentence.to_dict(self) for sentence in self.sentences]
        return researcher_dict
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def ask_gpt_query(self, query):
        with open("query_graph/gpt_prompts.json", "r") as f:
            prompt = json.loads(f.read())["initial prompt"]
        prompt += query
        response = callGPT(prompt)
        return response
    
    def get_attentions(self, sentences_list):
        headers = {
            'Authorization': f'Bearer {config.HF_read_APIKEY()}',
            'Content-Type': 'application/json',
        }

        json_data = {
            'inputs': sentences_list,
        }

        response = requests.post(config.attention_bert_endpoint(), headers=headers, json=json_data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Hugging Face request failed with status code: {response.status_code}")
        
    def get_k_search_queries(self, query_sentences, gpt_sentences, k, attention_threshold):
        combinations = []
        for query_sentence in query_sentences:
            for gpt_sentence in gpt_sentences:
                combinations.append(query_sentence + gpt_sentence)
        
        hf_response = self.get_attentions(combinations)
        
        attention_to_token = {} # keys
        highly_attended_from_token = {} # values / groups
        for comb_i, combination in enumerate(hf_response):
            attention_matrix = np.array(combination["average_attention"])
            tokens = combination["tokenized_input"]
            
            attention_to_j = attention_matrix.sum(axis=0, keepdims=False)
            for j in range(len(attention_to_j)):
                attention_to_token[(comb_i, j)] = attention_to_j[j]

            mean = attention_matrix.mean(axis=1, keepdims=False)
            std = attention_matrix.std(axis=1, keepdims=False)
            highly_attended_from_token |= {
                (comb_i, i): {
                    tokens[j]: ""
                    for j in range(attention_matrix.shape[1])
                    if attention_matrix[i, j] > mean[i]+std[i]*attention_threshold
            }
                for i in range(attention_matrix.shape[0])
                
            }

        top_k_attended = sorted(attention_to_token, key=lambda tok: attention_to_token[tok], reverse=True)[:k]
        search_queries = set()
        for attender in top_k_attended:
            query = " AND ".join(highly_attended_from_token[attender])    
            if len(search_queries) >= k:
                break
            search_queries.add(query)

        return list(search_queries)


    def get_k_urls(self, search_query, k):
        processed_urls_set = 'processed_urls'  # Redis key for the set

        current_results = 0
        start_index = 1
        fetched_urls = []

        while current_results < k:
            try:
                service = build("customsearch", "v1", developerKey=config.GGLSEARCH_APIKEY())
                res = service.cse().list(
                    q=search_query,
                    cx=config.GGL_SE(),
                    hl="en",
                    lr="lang_en",
                    start=start_index,
                    num=min(k-current_results, 10)
                ).execute()

                if 'items' in res:
                    for item in res['items']:
                        url = item['link']
                        if not redis_client.sismember(processed_urls_set, url):  # Check if URL is new
                            fetched_urls.append(url)
                            redis_client.sadd(processed_urls_set, url)  # Add URL to processed set
                            current_results += 1
                            if current_results >= k:
                                break

                start_index += 10

            except HttpError as e:
                print(f"An error has occurred: {e}")
                break
            except Exception as e:
                print(f"An unexpected error has occurred: {e}")
                break

        return fetched_urls

    
    def get_content_from_urls(self, urls, maximum_content=1000):
        sentences = []
        for url in urls:
            page = Page(url, self.context_window)
            for sentence in page.sentence_to_context:
                if len(sentences) >= maximum_content:
                    break
                sentences.append(Sentence(
                    sentence,
                    page.sentence_to_context[sentence], # context
                    url
                ))
        return sentences

    def embed_sentences(self, sentences):
        response = openai.Embedding.create(
            input=sentences,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']


    def get_top_n_sents(self):
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device="cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
        with torch.no_grad():
            researcher.gpt_response_embedding = model.encode(researcher.gpt_response)
            researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)

        # url to content in researcher.sentences
        logger.info(f"Collected {n_sentences_total} sentences")
        logger.debug("batching sentence embeddings")
        start = time.perf_counter()

        def sentence_generator(sentences, batch_size):
            batch = []
            for i in range(0, len(sentences), batch_size):
                if i + batch_size < len(sentences):
                    yield sentences[i:i+batch_size]
                else:
                    yield sentences[i:]
    

        with torch.no_grad():
            all_embeddings = []
            for batch_sentences in tqdm(sentence_generator(researcher.sentences, BATCH_SIZE), desc="batching sentence embeddings"):
                embeddings = model.encode(
                    list(sentence.text for sentence in batch_sentences),
                    batch_size=len(batch_sentences),
                    show_progress_bar=False,
                    device="cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu"
                )
                all_embeddings.append(embeddings)

            all_relevancies = []
            for i in tqdm(range(0, len(all_embeddings)), desc="batching sentence relevancy"):
                relevancies = util.cos_sim(
                    all_embeddings[i],
                    researcher.gpt_response_embedding
                )
                all_relevancies.append(relevancies)
            finish = time.perf_counter()
            logger.info(f"finished batching sentence embeddings and relevancies in {finish-start} seconds")
        

        # assign the vectorization and relevancy attributes to each sentence
        for sentence in researcher.sentences:
            i, j  = n_sents // BATCH_SIZE, n_sents % BATCH_SIZE
            sentence.embedding = all_embeddings[i][j]
            sentence.relevance = all_relevancies[i][j].item()
            n_sents += 1

        researcher.sentences = sorted(researcher.sentences, key=lambda s: -s.relevance)[:min(n_sents, len(researcher.sentences))]
        return researcher.sentences
    


class Page():
    timeout = 5
    max_sentences = 800

    def __init__(self, url, context_window): #content, url, ranking):
        self.url = url
        self.context_window = context_window

        self.content = self.get_webpage_content()
        if self.content:
            self.sentence_to_context = self.get_sentence_to_context(self.content, self.context_window)
        else:
            self.sentence_to_context = {}
        

    def get_webpage_content(self):
        logger.debug(f"getting content from {self.url}")
        try:
            # Send a GET request to the specified URL
            response = requests.get(self.url, timeout=self.timeout)
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
    
    def get_sentence_to_context(self, text, context_window):
        """_summary_

        Args:
            text (string): _description_
            context_window (int): fetching context_window sentences around the sentence

        Returns:
            dict: mapping sentence (key) to context (value). Both are strings
        """
        sentences = self.split_into_sentences(text)

        sent_to_context = {}
        for position, sentence in enumerate(sentences):
            if sentence not in sent_to_context: # assumption: duplicate sentences are web scraping error and not intentional
                sent_to_context[sentence] = self.get_sentence_content(position, context_window, sentences)

        return sent_to_context

    def split_into_sentences(self, text):
        # Split by paragraphs or newline characters as a basic method
        chunks = text.split('\n')
        sents = [sentence.strip() for chunk in chunks for sentence in sent_tokenize(chunk)]
        if len(sents) > Page.max_sentences:
            logger.debug(f"{url} has more than 800 sentences. Truncating to {Page.max_sentences} sentences")
            sents = sents[:Page.max_sentences]
        return sents

    
    def get_sentence_content(self, position, context_window, sentences):
        text = sentences[position]
        pre_context, post_context = "", ""
        if position > 0 and context_window > 0:
                pre_context = " ".join(sentences[max(0, position-context_window):position]).strip()
        if position < len(sentences) - 1 and context_window > 0:
            post_context = " ".join(sentences[position+1:min(len(sentences)-1, position+context_window+1)]).strip()
        return pre_context + " " + text + " " + post_context
    
class Sentence:
    def __init__(self, sentence, context, url):
        self.text = sentence
        self.context = context
        self.url = url
    
    def to_dict(self, researcher):
        sentence_dict = {
            'text': self.text,
            'context': self.context,
            'url': self.url,
            'search_queries': list(researcher.url_to_queries[self.url]),
        }
        if hasattr(self, 'relevance'):
            sentence_dict['relevance'] = self.relevance

        if hasattr(self, 'relevant_sentences'):
            sentence_dict['relevant_sentences'] = self.relevant_sentences

        if hasattr(self, 'neutrality'):
            sentence_dict['neutrality'] = self.neutrality
            sentence_dict["contradiction"] = self.contradiction
            sentence_dict["entailment"] = self.entailment

        return sentence_dict

    @classmethod
    def from_dict(cls, d):
        # Extracting main attributes
        sentence = cls(d['text'], d['context'], d['url'])

        # Setting optional attributes if they exist in the dictionary
        for attr in ['relevance', 'relevant_sentences', 'neutrality', 'contradiction', 'entailment']:
            if attr in d:
                setattr(sentence, attr, d[attr])

        return sentence
