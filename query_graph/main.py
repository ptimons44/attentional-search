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


# from logger import logger
# from researcher import Researcher
# use absolute imports for custom modules
from query_graph.logger import logger
from query_graph.researcher import Researcher


# def get_urls(researcher, parallelize_get_urls=True):
#     ### parallelized url retrevial
#     logger.debug("starting URL retrevial")
#     manager = Manager()
#     url_dict = manager.dict()

#     # Run the parallel jobs, passing the shared dictionary as an argument
#     start_time = time.perf_counter()
#     if parallelize_get_urls:
#         Parallel(n_jobs=-1, backend="loky")(delayed(researcher.get_k_urls)(search_query, url_dict, researcher.results_per_search) for search_query in researcher.search_queries)
#     else:
#         for search_query in researcher.search_queries:
#             researcher.get_k_urls(search_query, url_dict, researcher.results_per_search)

#     finish_time = time.perf_counter()
#     researcher.urls = dict(url_dict)
#     del url_dict

#     logger.debug(f"Gathered URL's in {finish_time-start_time} seconds")

def get_urls(researcher, parallelize_get_urls=True):
    logger.debug("starting URL retrieval")
    with Manager() as manager:
        url_to_queries = manager.dict()
        query_to_urls = manager.dict()

        start_time = time.perf_counter()

        if parallelize_get_urls:
            Parallel(n_jobs=-1, backend="threading")(
                delayed(researcher.get_k_urls)(search_query, researcher.results_per_search, url_to_queries, query_to_urls) 
                for search_query in researcher.search_queries
            )
        else:
            for search_query in researcher.search_queries:
                researcher.get_k_urls(search_query, researcher.results_per_search, url_to_queries, query_to_urls)

        finish_time = time.perf_counter()

        researcher.urls = dict(url_to_queries)
        researcher.query_to_urls = dict(query_to_urls)

        logger.debug(f"Gathered URLs in {finish_time-start_time} seconds")

def get_relevance(sentence):
    return -sentence.relevance

def get_sentences_and_pages(researcher, model, parallelize_create_pages=True, BATCH_SIZE = 512):
    ### parallelized page creation
    logger.debug("starting page and sentences creation")
    manager = Manager()
    sentences_list = manager.list([])

    # Run the parallel jobs, passing the shared dictionary as an argument # search_queries, url, pages_dict
    start_time = time.perf_counter()
    if parallelize_create_pages:
        # create_pages_and_sentences(self, search_queries, url, sentence_list, model)
        Parallel(n_jobs=4, backend="loky")(delayed(researcher.create_pages_and_sentences)(researcher.urls[url], url, sentences_list) for url in researcher.urls)
    else:
        for url in researcher.urls:
            researcher.create_pages_and_sentences(researcher.urls[url], url, sentences_list)
    # HUNG
    finish_time = time.perf_counter()
    logger.debug(f"Created sentences in {finish_time-start_time} seconds")

    logger.debug("batching sentence embeddings")
    logger.info(f"Collected {len(sentences_list)} sentences")
    print(f"Batching sentence embeddings for {len(sentences_list)} sentences")
    start = time.perf_counter()

    def sentence_generator(sentences_list, batch_size):
        for i in range(0, len(sentences_list), batch_size):
            yield sentences_list[i:i + batch_size]

    all_embeddings = []
    for batch_sentences in tqdm(sentence_generator(sentences_list, BATCH_SIZE), desc="batching sentence embeddings"):
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
    logger.debug(f"finished batching sentence embeddings and relevancies in {finish-start} seconds")

    researcher.nodes = list(sentences_list) # reassign Sentence objects to researcher.nodes
    for sentence in researcher.nodes:
        i, j  = sentence.index // BATCH_SIZE, sentence.index % BATCH_SIZE
        sentence.embedding = all_embeddings[i][j]
        sentence.relevance = all_relevancies[i][j].item()


    logger.debug("sorting sentences")
    start_time = time.perf_counter()
    researcher.nodes.sort(key=get_relevance) # lambda function causes pickling error
    researcher.nodes = researcher.nodes[:min(researcher.num_nodes, len(researcher.nodes))]
    # del sentences_list
    finish_time = time.perf_counter()
    logger.debug(f"sorted nodes by similarity in {finish_time-start_time} seconds")

def pipeline(query, num_nodes=50, parallelize_get_urls=True, parallelize_create_pages=True):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device="cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
    nlp = spacy.load("en_core_web_sm")
    researcher = Researcher(query, nlp=nlp, num_nodes=num_nodes)
    researcher.gpt_response_embedding = model.encode(researcher.gpt_response)
    researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)
    print("ChatGPT: " + researcher.gpt_response)

    get_urls(researcher, parallelize_get_urls=parallelize_get_urls)
    get_sentences_and_pages(researcher, model, parallelize_create_pages=parallelize_create_pages)
    
    # calculate which gpt sentences are most similar to each node
    researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)
    for node in researcher.nodes:
        node.gpt_relevance_by_sentence = util.cos_sim(
            node.embedding,
            researcher.gpt_sentences_embedding
        )
        mean = node.gpt_relevance_by_sentence.mean()
        std = node.gpt_relevance_by_sentence.std()
        node.relevant_sentences = node.gpt_relevance_by_sentence > mean + std

    # return "\n\n".join([node.text for node in researcher.nodes])
    return researcher
    
def ordinal(n):
    if str(n)[-1] == '1':
        return str(n) + 'st'
    elif str(n)[-1] == '2':
        return str(n) + 'nd'
    elif str(n)[-1] == '3':
        return str(n) + 'rd'
    else:
        return str(n) + 'th'
    
import dill

def serialize_object(obj, filename):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)

def deserialize_object(filename):
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    return obj

if __name__ == "__main__":
    filename = "researcher.pkl"


if __name__ == "__main__":
    start = time.perf_counter()
    query = input("Enter query: ")
    if query == "":
        query = "Is it true that Neil Armstrong never went to space and that he was a paid actor by the inner circle, AKA NASA?"
    logger.info("starting new run with query: " + query)
    
    num_nodes = input("Enter number of nodes (usually on order of 25-150): ")
    try:
        num_nodes = int(num_nodes)
    except:
        num_nodes = 25
    print("Asking ChatGPT for response...")
    researcher = pipeline(query, num_nodes=num_nodes, parallelize_create_pages=True, parallelize_get_urls=False)
    # serialize_object(researcher, "researcher.pkl")
    # researcher = deserialize_object("researcher.pkl")
    # raise AssertionError
    end = time.perf_counter()
    logger.info(f"finished in {end-start} seconds")

    # output
    print("\n\nContent from the Web (obtained via Google Search):")
    for (index, node) in enumerate(researcher.nodes):
        # print(node.relation_to_gpt)
        print(f"{ordinal(index+1)} most similar sentence to ChatGPT's response. Relevance to ChatGPT's response: ", node.relevance)
        print("sentence: ", node.text)
        print("context: ", node.context)
        print("Most relevant ChatGPT sentences:")
        print([researcher.gpt_sentences[idx] for (idx, val) in enumerate(node.relevant_sentences[0]) if val.item()])
        print("url", node.url)
        print("search queries", node.search_queries)
        print("\n\n")