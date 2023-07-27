from query_graph.researcher import Researcher
from query_graph.gpt import embed_sentences

from celerysetup import celery_app

from sentence_transformers import util

@celery_app.task
def init_researcher(query, n_sents=100):
    researcher = Researcher(query, n_sents=n_sents)
    return researcher.to_dict()

@celery_app.task
def probe_search_query(researcher_dict, search_query):
    researcher = Researcher.from_dict(researcher_dict)
    urls = researcher.get_k_urls(search_query, researcher.results_per_search)
    researcher.query_to_urls[search_query] = urls
    for url in urls:
        if url in researcher.url_to_queries:
            researcher.url_to_queries[url].append(search_query)
        else:
            researcher.url_to_queries[url] = [search_query]

    sentences = researcher.get_content_from_urls(urls)
    
    embeddings = embed_sentences(list(s.text for s in sentences))
    if not hasattr(researcher, 'gpt_response_embedding'):
        researcher.gpt_response_embedding = embed_sentences([researcher.gpt_response])
    relevancies = util.cos_sim(
                    embeddings,
                    researcher.gpt_response_embedding
                )
    for index, sentence in enumerate(sentences):
        sentence.relevance = relevancies[index].item()
    if hasattr(researcher, 'sentences'):
        researcher.sentences.extend(sentences)
    else:
        researcher.sentences = sentences
    researcher.sentences.sort(key=lambda s: s.relevance, reverse=True)
    
    return researcher.to_dict()


@celery_app.task
def debug_task():
    print('Debug task!')
    return 'Debug task!'
