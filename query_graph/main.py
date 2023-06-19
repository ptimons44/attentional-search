import query_parser

def create_query_graph(query):
    """returns the query graph associated for a particular input query

    Args:
        query (string): user's (textual) query, such as a question or research topic
    """
    search_queries = query_parser.generate_search_queries(query)
    nodes = dict()
    for search_query in search_queries:
        scraped_content = scrape_content(search_query)
        relevant_excerpts = find_relevant_content(scraped_content)
        nodes |= relevant_excerpts