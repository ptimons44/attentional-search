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
    researcher = Researcher.Researcher(search_queries, num_search_results)


    def create_page(search_query, url, pages):
        page = Researcher.Page(search_query,url)
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

    # print("maping:\n", gpt_keyword_sentence_mapping)
    # print("search_queries", search_queries)
    # content = dict()
    # for search_query in search_queries:
    #     scraped_content = get_top_k_content(search_query, content, k=num_search_results)
    #     content |= scraped_content
    # with open("temp.json", "w") as f:
    #     json_object = json.dumps(content, indent=4)
    #     f.write(json_object)
    # #     return
    # with open("temp.json", "r") as f:
    #     content = json.load(f)
    # nodes = get_k_most_similar_sents(content, query, k=num_nodes, context_window=context_window)

    # classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')

    # nlp = spacy.load("en_core_web_sm")
    # gpt_sentences = list(nlp(gpt_response).sents)

    # graph = dict()
    # for ix, (similarity,source,sent,context,search_phrase) in enumerate(nodes):
    #     # try:
    #     print("keyword extracted: ", gpt_keyword_from_query(search_phrase))
    #     sentence_index = gpt_keyword_sentence_mapping[gpt_keyword_from_query(search_phrase)]
    #     print("sentence index", sentence_index)
    #     print("sentence: ", gpt_sentences[sentence_index])
    #     relation = classify_relationship(sent, gpt_sentences[sentence_index], classifier)
    #     print("relation", relation)
    #     graph[context] = (relation, gpt_sentences[sentence_index])

    # except:
    #         print("RegexError: GPT keyword not found")


    # return graph

if __name__ == "__main__":
    nlp = spacy.load(config.language_model())
    # query = "Was the 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots?"
    query = "Was Issac Newton the first person to discover gravity??"
    graph = create_query_graph(query, 3, nlp)
    for node in graph:
        print("web sentence:", node)
        print(graph[node])
        print("\n")
    # for ix, (similarity,source,sent,context,search_phrase) in enumerate(result):
    #     print(ix, ".", search_phrase, ":", context)
    #     print("\n\n")