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

def get_sentences_from_queries(researcher, model, BATCH_SIZE=512):
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
    
    logger.debug("sorting sentences")
    start_time = time.perf_counter()
    researcher.sentences.sort(key=get_relevance) # lambda function causes pickling error
    finish_time = time.perf_counter()
    logger.debug(f"sorted sentences by similarity in {finish_time-start_time} seconds")

def get_low_dim_embedding(researcher, model, n_sents=300, batch_size=16):
    print("reload sucessful")
    # calculate which gpt sentences are most similar to each sentence
    researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)
    for sentence in researcher.sentences[:min(n_sents, len(researcher.sentences))]:
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
    
    # calculate the ['entails', 'contradicts', 'neutral'] score of the top n_sents sentences

    # Initialize the model and tokenizer
    model_name = 'roberta-large-mnli'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create all pair combinations
    sentence_pairs = []
    for sentence in researcher.sentences[:min(n_sents, len(researcher.sentences))]:
        sentence.contradiction, sentence.neutrality, sentence.entailment = [], [], []
        for relevant in sentence.relevant_sentences:
            sentence_pairs.append((sentence, researcher.gpt_sentences[relevant]))

    # Create batches
    batches = np.array_split(sentence_pairs, len(sentence_pairs) // batch_size)

    for batch in batches:
        # Prepare batch data
        batch_sentences1 = [pair[0].text for pair in batch]
        batch_sentences2 = [pair[1] for pair in batch]

        # Encode the sentences
        encoded_input = tokenizer(batch_sentences1, batch_sentences2, padding=True, truncation=True, max_length=128, return_tensors='pt')

        # Get model output
        output = model(**encoded_input)
        logits = output.logits

        # Apply softmax function to transform logits into probabilities
        probabilities = F.softmax(logits, dim=-1).detach().numpy()

        # store results in sentence's entails, contradicts, neutral attributes
        for pair, probs in zip(batch, probabilities):
            pair[0].contradiction.append(probs[0])
            pair[0].neutrality.append(probs[1])
            pair[0].entailment.append(probs[2])

def researcher_to_df(researcher, n_sents):
    # Create an empty list to store dictionaries of sentence attributes
    sentence_list = []

    # Iterate over each sentence in the researcher object
    for sentence in researcher.sentences[:min(n_sents, len(researcher.sentences))]:
        # Create a dictionary of s attributes
        sentence_dict = {
            'text': sentence.text,
            'context': sentence.context,
            'url': sentence.url,
            'search_queries': researcher.urls[sentence.url],
            'relevant_sentences': sentence.relevant_sentences,
            'contradiction': sentence.contradiction,
            'neutrality': sentence.neutrality,
            'entailment': sentence.entailment,
            'relevance': sentence.relevance
        }

        # Append the sentence dictionary to the list
        sentence_list.append(sentence_dict)

    # Create a DataFrame from the list of sentence dictionaries
    df = pd.DataFrame(sentence_list)
    return df

def researcher_to_dict(researcher, n_sents):
    # Create an empty list to store dictionaries of sentence attributes
    sentences_dict = {}

    # Iterate over each sentence in the researcher object
    for (i, sentence) in enumerate(researcher.sentences[:min(n_sents, len(researcher.sentences))]):
        # Create a dictionary of s attributes
        sentence_dict = {
            'text': sentence.text,
            'context': sentence.context,
            'url': sentence.url,
            'search_queries': list(researcher.urls[sentence.url]),
            'relevant_sentences': sentence.relevant_sentences,
            'contradiction': sentence.contradiction,
            'neutrality': sentence.neutrality,
            'entailment': sentence.entailment,
            'relevance': sentence.relevance
        }

        # Append the sentence dictionary to the list
        sentences_dict[i] = sentence_dict

    # Create a DataFrame from the list of sentence dictionaries
    return sentences_dict

def get_llm_response(query, n_sents=300):
    researcher = Researcher(query, num_nodes=n_sents)
    return researcher

def get_web_content(researcher, n_sents=300, parallelize_get_urls=True, parallelize_create_pages=True):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device="cuda" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
    researcher.gpt_response_embedding = model.encode(researcher.gpt_response)
    researcher.gpt_sentences_embedding = model.encode(researcher.gpt_sentences)

    get_urls(researcher, parallelize_get_urls=parallelize_get_urls)
    get_sentences_from_queries(researcher, model)
    get_low_dim_embedding(researcher, model, n_sents=n_sents)
    researcher_df = researcher_to_df(researcher, n_sents=n_sents)

    return researcher_df
    
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