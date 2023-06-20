from parsing import generate_search_queries
from scraping import get_top_k_content
from extraction import get_k_most_similar_sents

import json

def create_query_graph(query, num_search_results, num_nodes=100):
    """returns the query graph associated for a particular input query

    Args:
        query (string): user's (textual) query, such as a question or research topic
    """
    search_queries = generate_search_queries(query)
    content = dict()
    for search_query in search_queries:
        scraped_content = get_top_k_content(search_query, k=num_search_results)
        content |= scraped_content
    nodes = get_k_most_similar_sents(content, query, k=num_nodes)
    return nodes

if __name__ == "__main__":
    query = "The 2020 election was stolen from Trump because of fraudulent voting machines and electronic ballots."
    res = create_query_graph(query, 3)
    print(res)