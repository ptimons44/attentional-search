from query_graph.researcher import Researcher
from query_graph.gpt import embed_sentences

from celerysetup import celery_app, logger

from sentence_transformers import util

@celery_app.task
def init_researcher(query, n_sents=100):
    researcher = Researcher(query, n_sents=n_sents)
    return researcher.to_dict()

import joblib
@celery_app.task
def query_to_sentences(researcher_dict, search_query):
    logger.info("query_to_sentences called")
    try:
        data = joblib.load('cache/all_sentences.joblib')
        logger.info(f"using cache for query: {search_query}")
        return list(sentence for sentence in data if sentence["search_query"] == search_query)
    except:
        logger.info(f"not using cache for query: {search_query}")

        researcher = Researcher.from_dict(researcher_dict)
        urls = researcher.get_k_urls(search_query, researcher.results_per_search)

        sentences = researcher.get_content_from_urls(urls, search_query)
        
        embeddings = embed_sentences(list(s.text for s in sentences))
        relevancies = util.cos_sim(
                        embeddings,
                        researcher.gpt_response_embedding
                    )
        for index, sentence in enumerate(sentences):
            sentence.relevance = relevancies[index].item()
            sentence.embedding = embeddings[index]
        sentences.sort(key=lambda s: s.relevance, reverse=True)
        return list(s.to_dict() for s in sentences)


from sklearn.manifold import TSNE

# Fit t-SNE to your data
tsne_model = TSNE(n_components=2, random_state=0)  # Use n_components=3 for 3D

import numpy as np

@celery_app.task
def compress_sentences(sentences):
    embeddings = np.array(list(np.array(s["embedding"]) for s in sentences))
    reduced_data = tsne_model.fit_transform(embeddings)
    return reduced_data.tolist()