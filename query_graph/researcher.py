# TODO: remove when ready to deploy
# import sys
# sys.path.insert(0, "/Users/patricktimons/Documents/GitHub/query-graph")

from nltk import sent_tokenize
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
model_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(model_name, output_attentions=True)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)

import numpy as np

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

# using absolute imports for custom modules
from query_graph.gpt import callGPT
from query_graph import config
from query_graph.logger import logger

from redis import Redis
redis_client = Redis()


    
class Parser:
    def __init__(self, researcher):
        self.researcher = researcher
        # TODO: uncomment when debugging is complete
        self.search_queries = self.get_k_search_queries(
            self.researcher.query_sentences, 
            self.researcher.gpt_sentences,
            self.researcher.num_search_queries,
            self.researcher.search_query_attention_threshold
        )

    def get_token_idx_to_word(self, token_ids):
        # dictionary mapping token index to (corresponding word, is_child_root) 
        # parent root is for tracking the chain of updates that we will have to do
        token_idx_to_word = {}
        for idx, token in enumerate(token_ids):
            decoded_token = "".join(bert_tokenizer.decode(token).split())
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
    
    def get_k_search_queries(self, query_sentences, gpt_sentences, k, attention_threshold):
        print("getting k search queries")
        # tokens represented as (query_sentence_idx, gpt_sentence_idx, token_idx)
        attention_to_token = {} # keys
        highly_attended_from_token = {} # values / groups

        token_to_word = {}
        ignore_token = {}

        # ignore stop words, punctuation, special tokens, and lone numerals
        ignore = set(stopwords.words('english')) | set(string.punctuation) | {'[CLS]', '[SEP]'} | set("0123456789") 

        for (q_idx, query_sentence) in enumerate(query_sentences):
            for (gpt_idx, gpt_sentence) in enumerate(gpt_sentences):
                print(f"getting attention for query sentence {q_idx} and gpt sentence {gpt_idx}")
                inputs = bert_tokenizer.encode_plus(query_sentence, gpt_sentence, return_tensors='pt')
                print(f"Inputs tokenized for query {q_idx} and gpt {gpt_idx}")
        
                # Get token ids for decoding back to words later
                token_ids = inputs['input_ids'].numpy().squeeze()
                print(f"Token ids retrieved for query {q_idx} and gpt {gpt_idx}")
        
                # dictionary mapping token index to (corresponding word, is_child_root) 
                # parent root is for tracking the chain of updates that we will have to do
                token_idx_to_word = self.get_token_idx_to_word(token_ids, bert_tokenizer)
                print(f"Token idx to word mapping done for query {q_idx} and gpt {gpt_idx}")
        
                token_to_word |= {(q_idx, gpt_idx, token_idx): token_idx_to_word[token_idx] for token_idx in token_idx_to_word}
                print(f"Token to word dictionary updated for query {q_idx} and gpt {gpt_idx}")
        
                outputs = bert_model(**inputs)
                print(f"Model outputs retrieved for query {q_idx} and gpt {gpt_idx}")
        
                last_layer_attentions = outputs.attentions[-1]  # get the last layer's attentions
                avg_attention = last_layer_attentions.squeeze(0).mean(dim=0)  # average attention across heads
                attention_matrix = avg_attention.detach().numpy()
                print(f"Attention matrix calculated for query {q_idx} and gpt {gpt_idx}")
        
                keep_token = np.array([bool(bert_tokenizer.decode([token_ids[idx]]) not in ignore) for idx in range(attention_matrix.shape[0])])
                print(f"Tokens to keep identified for query {q_idx} and gpt {gpt_idx}")

                attention_to_j = attention_matrix.sum(axis=0, keepdims=False)
                for j in range(len(attention_to_j)):
                    if keep_token[j]:
                        attention_to_token[(q_idx, gpt_idx, j)] = attention_to_j[j]
                    ignore_token[(q_idx, gpt_idx, j)] = not keep_token[j]
                print(f"Attention to token dictionary updated for query {q_idx} and gpt {gpt_idx}")
        
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
                print(f"Highly attended tokens identified for query {q_idx} and gpt {gpt_idx}")
        print("here 1")
        def attention_to_token_key(token):
            if ignore_token[token]:
                return 0
            else:
                return attention_to_token[token]
        top_k_attendees = sorted(attention_to_token, key=attention_to_token_key, reverse=True)[:k]
        
        print("here 2")

        search_queries = set()
        for attender in top_k_attendees:
            query = " AND ".join(word for word in highly_attended_from_token[attender])
            if len(search_queries) >= k:
                break
            search_queries.add(query)
        print("here 3")
        return list(search_queries)


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
            parser = Parser(self)
            self.search_queries = parser.search_queries
            logger.info(f"Researcher initialized with the following search queries: {self.search_queries}")
        

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

    
    # def get_k_urls(self, search_query, k, url_to_queries, query_to_urls):
    #     current_results = 0
    #     start_index = 1
    #     while current_results < k:
    #         try:
    #             service = build("customsearch", "v1", developerKey=config.GGLSEARCH_APIKEY())
    #             res = service.cse().list(
    #                 q=search_query,
    #                 cx=config.GGL_SE(),
    #                 hl="en",
    #                 lr="lang_en",
    #                 start=start_index,
    #                 num=min(k-current_results, 10), # Requesting the minimum between remaining results and maximum results per page (10)
    #             ).execute()
    #             if 'items' in res:
    #                 for item in res['items']:
    #                     url = item['link']
    #                     with lock: # Adding thread safety
    #                         if (current_results < k) and (url not in url_to_queries):
    #                             url_to_queries[url] = {search_query}
    #                             current_results += 1
    #                         else:
    #                             query_set = set(url_to_queries[url])
    #                             query_set.add(search_query)
    #                             url_to_queries[url] = query_set

    #                         if search_query not in query_to_urls:
    #                             query_to_urls[search_query] = [url]
    #                         else:
    #                             # create a regular Python list, update it, and then assign it back to the Manager dictionary
    #                             url_list = list(query_to_urls[search_query])
    #                             url_list.append(url)
    #                             query_to_urls[search_query] = url_list

    #             start_index += 10 # Update starting index for next results page
    #         except HttpError as e:
    #             print(f"An error has occurred: {e}")
    #             break
    #         except Exception as e:
    #             print(f"An unexpected error has occurred: {e}")
    #             break

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


    # def get_content_from_query(self, search_query, maximum_content=1000):
    #     logger.debug(f"scraping web with search query \"{search_query}\"")
    #     content_amount = 0
    #     for url in self.query_to_urls[search_query]:
    #         sentences = []
    #         if url not in self.cache:
    #             logger.debug(f"creating page and sentences for {url}")
    #             page = Page(url, self.context_window)
    #             for sentence in page.sentence_to_context:
    #                 sentences.append(
    #                     Sentence(
    #                         sentence,
    #                         page.sentence_to_context[sentence], # context
    #                     )
    #                 )
    #             with lock:
    #                 self.cache[url] = sentences[:min(len(sentences), maximum_content-content_amount)]
    #                 content_amount += len(sentences)

                
    #         if content_amount >= maximum_content:
    #             break
    #     logger.debug(f"search query \"{search_query}\" scraped {min(content_amount, maximum_content)} sentences")
    #     return content_amount

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
            'search_queries': list(researcher.urls[self.url]),
            'relevant_sentences': self.relevant_sentences,
            'relevance': self.relevance
        }
        if hasattr(self, 'neutrality'):
            sentence_dict['neutrality'] = self.neutrality
            sentence_dict["contradiction"] = self.contradiction
            sentence_dict["entailment"] = self.entailment

        return sentence_dict
