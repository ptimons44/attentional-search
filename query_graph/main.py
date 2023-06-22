from information_extraction.parsing import generate_search_queries
from information_extraction.scraping import get_top_k_content
from information_extraction.extraction import get_k_most_similar_sents
from information_extraction.gpt import callGPT

import json
import spacy
import config

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
    gpt_respone = ask_gpt_query(query)
    search_queries = generate_search_queries(query, gpt_respone, nlp)
    content = dict()
    for search_query in search_queries:
        scraped_content = get_top_k_content(search_query, content, k=num_search_results)
        content |= scraped_content
    # with open("temp.json", "w") as f:
    #     json_object = json.dumps(content, indent=4)
    #     f.write(json_object)
    # #     return
    # with open("temp.json", "r") as f:
    #     content = json.load(f)
    nodes = get_k_most_similar_sents(content, query, k=num_nodes, context_window=context_window)

    return nodes

if __name__ == "__main__":
    nlp = spacy.load(config.language_model())
    # query = "Was the 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots?"
    query = "Is climate change a timely threat to humanity?"
    result = create_query_graph(query, 3, nlp)
    for ix, (similarity,source,sent,context,search_phrase) in enumerate(result):
        print(ix, ".", search_phrase, ":", context)
        print("\n\n")