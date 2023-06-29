import time
from joblib import Parallel, delayed
from multiprocessing import Manager

import spacy

from researcher import Researcher

from sentence_transformers import SentenceTransformer, util

import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"

def get_relevance(sentence):
    return -sentence.relevance

def pipeline(query, parallelize_get_urls=True, parallelize_create_pages=True, parallelize_create_sentences=False):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    nlp = spacy.load("en_core_web_sm")
    researcher = Researcher(query, nlp=nlp)
    researcher.gpt_response_embedding = model.encode(researcher.gpt_response)
    
    ### parallelized url retrevial
    print("starting multiprocessing 1")
    manager = Manager()
    url_dict = manager.dict()

    # Run the parallel jobs, passing the shared dictionary as an argument
    start_time = time.perf_counter()
    # search_queries = researcher.search_queries
    if parallelize_get_urls:
        Parallel(n_jobs=-1, backend="loky")(delayed(researcher.get_urls)(search_query, url_dict) for search_query in researcher.search_queries)
    else:
        for search_query in researcher.search_queries:
            researcher.get_urls(search_query, url_dict)
    finish_time = time.perf_counter()
    researcher.urls = url_dict
    del url_dict

    print(f"Gathered URL's in {finish_time-start_time} seconds")


    ### parallelized page creation
    print("starting multiprocessing 2")
    manager = Manager()
    pages_dict = manager.dict()

    # Run the parallel jobs, passing the shared dictionary as an argument # search_queries, url, pages_dict
    start_time = time.perf_counter()
    if parallelize_create_pages:
        Parallel(n_jobs=4, backend="loky")(delayed(researcher.create_page)(researcher.urls[url], url, pages_dict) for url in researcher.urls)
    else:
        for url in researcher.urls:
            researcher.create_page(researcher.urls[url], url, pages_dict)
    finish_time = time.perf_counter()
    researcher.pages = pages_dict
    del pages_dict

    print(f"Gathered URL's in {finish_time-start_time} seconds")

    print("Getting nodes")
    manager = Manager()
    sentences_list = manager.list([])
    
    start_time = time.perf_counter()
    if parallelize_create_sentences:
        Parallel(n_jobs=4, backend="loky")(delayed(researcher.create_sentence)(
            researcher.pages[page].search_queries,
            sentence_text,
            researcher.pages[page].get_sentence_content(position, researcher.context_window),
            model,
            sentences_list
        )
            for page in researcher.pages
            if researcher.pages[page].content
            for position, sentence_text in enumerate(researcher.pages[page].sentences)
            )
    else:
        for page in researcher.pages:
            if researcher.pages[page].content:
                for (position, sentence_text) in enumerate(researcher.pages[page].sentences):
                    researcher.create_sentence(
                        researcher.pages[page].search_queries, # search_queries
                        sentence_text, # sentence
                        researcher.pages[page].get_sentence_content(position, researcher.context_window), # context
                        model, # model
                        sentences_list
                    )
    finish_time = time.perf_counter()
    print(f"created sentences in {finish_time-start_time} seconds")
    researcher.nodes = sentences_list
    del sentences_list

    print("sorting sentences")
    start_time = time.perf_counter()
    researcher.nodes.sort(key = get_relevance) # lambda function causes pickling error
    researcher.nodes = researcher.nodes[:min(researcher.num_nodes, len(researcher.nodes))]
    finish_time = time.perf_counter()
    print(f"sorted nodes by similarity in {finish_time-start_time} seconds")

    return researcher
    


    

if __name__ == "__main__":
    # query = "Was the 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots?"
    query = "Is it true that Issac Newton was the first person to discover gravity."
    # query = "Is it true that Neil Armstrong never went to space; he was a paid actor by the inner circle AKA NASA?"
    researcher = pipeline(query)


    # output
    print("\n\noutput:")
    end = time.time()
    for node in researcher.nodes:
        # print(node.relation_to_gpt)
        print(node.context)
        print(node.relevance)
        print("\n\n")

    # print(f"created researcher in {end-start} seconds")
    