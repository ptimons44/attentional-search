from information_extraction.parsing import generate_search_queries
# from information_extraction.scraping import get_top_k_content
# from information_extraction.extraction import get_k_most_similar_sents
from information_extraction.gpt import callGPT
# from comparasin.comparasin import classify_relationship

import json
import threading
import spacy
import config

import re 

from transformers import pipeline

import Researcher

def gpt_keyword_from_query(input_string):
    pattern = r'\b(?:AND|OR)\b\s+(.+)$'
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return None

def ask_gpt_query(query):
    with open("query_graph/information_extraction/gpt_prompts.json", "r") as f:
        prompt = json.loads(f.read())["initial prompt"]
    prompt += query
    response = callGPT(prompt)
    return response

def create_query_graph(query, num_search_results, nlp, num_nodes=100, context_window=1):
    """returns the query graph associated for a particular input query

    Args:
        query (string): user's (textual) query, such as a question or research topic
    """
    gpt_response = ask_gpt_query(query)
    search_queries, gpt_keyword_sentence_mapping = generate_search_queries(query, gpt_response, nlp)
    researcher = Researcher.Researcher(query, search_queries, gpt_response, num_search_results)
    nodes = researcher.top_k_similar_sentences()
    print("num nodes", len(nodes))
    print(nodes[0])
    for sentence in nodes:
        sentence.get_relation_to_query(researcher.query)
    return nodes

if __name__ == "__main__":
    nlp = spacy.load(config.language_model())
    query = "Was the 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots?"
    # query = "Issac Newton was the first person to discover gravity."
    # query = "Neil Armstrong never went to space; he was a paid actor by the inner circle AKA NASA."
    graph = create_query_graph(query, 3, nlp)
    for node in (graph):
        print(node.relation_to_query)
        print(node.context)
        print("\n\n")
    # search = Researcher.Search(query, 5)
    # print(search.search_google())