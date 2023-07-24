# TODO: delete when in produciton
import sys
sys.path.insert(0, "/Users/patricktimons/Documents/GitHub/query-graph")

import time
from tqdm import tqdm

from joblib import Parallel, delayed
from multiprocessing import Manager

import spacy

from sentence_transformers import SentenceTransformer, util

import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"

import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import numpy as np
import pandas as pd

# use absolute imports for custom modules
from query_graph.logger import logger
from query_graph.researcher import Researcher

def get_urls(researcher):
    logger.debug("starting URL retrieval")
    with Manager() as manager:
        url_to_queries = manager.dict()
        query_to_urls = manager.dict()

        start_time = time.perf_counter()

        Parallel(n_jobs=-1, backend="threading")(
            delayed(researcher.get_k_urls)(search_query, researcher.results_per_search, url_to_queries, query_to_urls) 
            for search_query in researcher.search_queries
        )

        finish_time = time.perf_counter()

        researcher.urls = dict(url_to_queries)
        researcher.query_to_urls = dict(query_to_urls)

        logger.debug(f"Gathered URLs in {finish_time-start_time} seconds")

def get_sentences_from_queries(researcher, model, n_sents=150, BATCH_SIZE=512):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Make a list of all your search queries
    search_queries = list(researcher.query_to_urls.keys())

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(researcher.get_content_from_query, search_query): search_query for search_query in search_queries}
        
        n_sentences_total = 0
        for future in as_completed(future_to_url):
            search_query = future_to_url[future]
            try:
                n_sentences_found = future.result()
                n_sentences_total += n_sentences_found
            except Exception as exc:
                print(f'{search_query} generated an exception: {exc}')
    
    # url to content in researcher.sentences
    logger.info(f"Collected {n_sentences_total} sentences")
    logger.debug("batching sentence embeddings")
    start = time.perf_counter()
    
    def sentence_generator(cache, batch_size):
        batch = []
        for (i, url) in enumerate(cache):
            batch += cache[url]
            if len(batch) >= batch_size:
                yield batch[:batch_size]
                batch = batch[batch_size:]
            elif i == len(cache) - 1:
                # no more urls
                yield batch

    with torch.no_grad():
        all_embeddings = []
        for batch_sentences in tqdm(sentence_generator(researcher.cache, BATCH_SIZE), desc="batching sentence embeddings"):
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
    # assign sentences to researcher
    researcher.sentences = []
    n_sents = 0
    for url in researcher.cache:
        for sentence in researcher.cache[url]:
            i, j  = n_sents // BATCH_SIZE, n_sents % BATCH_SIZE
            sentence.embedding = all_embeddings[i][j]
            sentence.relevance = all_relevancies[i][j].item()
            sentence.url = url
            researcher.sentences.append(sentence)
            n_sents += 1
    researcher.cache = {}

    researcher.sentences = sorted(researcher.sentences, key=lambda s: -s.relevance)[:min(n_sents, len(researcher.sentences))]

def get_relevant_gpt_sentences(researcher, model):
    """_summary_

    Args:
        researcher (_type_): _description_
        model (SentenceTransformer): _description_
    """
    # calculate which gpt sentences are most similar to each sentence
    with torch.no_grad():
        researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)
        for sentence in researcher.sentences:
            sentence.gpt_relevance_by_sentence = util.cos_sim(
                sentence.embedding,
                researcher.gpt_sentences_embedding
            )
            mean = sentence.gpt_relevance_by_sentence.mean()
            std = sentence.gpt_relevance_by_sentence.std()
            sentence.relevant_sentences = [
                idx 
                for idx in range(len(researcher.gpt_sentences)) 
                if sentence.gpt_relevance_by_sentence.squeeze()[idx].item() > mean + std
                ]
            
def gpt_sentence_3D(researcher, gpt_sentence, mnli_model, mnli_tokenizer, batch_size=16):
    """_summary_

    Args:
        researcher (_type_): _description_
        gpt_sentence (int): index of gpt sentence
        mnli_model (_type_): AutoModelForSequenceClassification
    """
    # Create all pair combinations
    sentence_pairs = []
    for sentence in researcher.sentences:
        if gpt_sentence in sentence.relevant_sentences:
            sentence_scores = {'contradiction': None, 'neutrality': None, 'entailment': None}
            sentence_pairs.append((
                sentence.text,
                researcher.gpt_sentences[gpt_sentence],
                sentence_scores
            ))
    
    # output
    graph = set()

    # Create batches
    batches = np.array_split(sentence_pairs, max(1, len(sentence_pairs) // batch_size))
    print("batches", batches)
    for batch in batches:
        # Prepare batch data
        batch_sentences1 = [pair[0] for pair in batch]
        batch_sentences2 = [pair[1] for pair in batch]

        # Encode the sentences
        encoded_input = mnli_tokenizer(batch_sentences1, batch_sentences2, padding=True, truncation=True, max_length=128, return_tensors='pt')

        # Get model output
        output = mnli_model(**encoded_input)
        logits = output.logits
        print("logits", logits)

        # Apply softmax function to transform logits into probabilities
        probabilities = F.softmax(logits, dim=-1).detach().numpy()
        print("probabilities", probabilities)

        sentences = {
            {'text': pair[0],
            'contradiction': prob[0],
            'neutrality': prob[1],
            'entailment': prob[2]
            }
            for pair, prob in zip(batch, probabilities)
        }

        graph.union(sentences)
        return graph

def all_gpt_sentence_3D(researcher):
    # Initialize the model and tokenizer
    model_name = 'roberta-large-mnli'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    graphs_3D = {}
    
    for gpt_sentence in range(len(researcher.gpt_sentences)):
        graph = gpt_sentence_3D(researcher, gpt_sentence, model, tokenizer)
        graphs_3D[gpt_sentence] = graph

    return graphs_3D


def get_llm_response(query, n_sents=150):
    researcher = Researcher(query, num_nodes=n_sents)
    return researcher.to_dict()

def get_web_content(researcher_dict, n_sents=150):
    researcher = Researcher.from_dict(researcher_dict)
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device="cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
    with torch.no_grad():
        researcher.gpt_response_embedding = model.encode(researcher.gpt_response)
        researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)

    get_urls(researcher)
    get_sentences_from_queries(researcher, model)
    get_relevant_gpt_sentences(researcher, model)
    return researcher.to_dict()