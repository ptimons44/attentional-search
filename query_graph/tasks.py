import os
import joblib
from query_graph.researcher import Researcher
from query_graph.gpt import embed_sentences

from celerysetup import celery_app, logger
from sentence_transformers import util
import numpy as np
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=3, random_state=0)  # Use n_components=3 for 3D

cache_results = False
load_from_cache = True
cache_dir = "cache/55b509b8-761d-4998-b850-f57403532d7a"

@celery_app.task(bind=True)
def init_researcher(self, query, n_search_queries, n_sents=100):
    if load_from_cache:
        return joblib.load(f"{cache_dir}/researcher_result_.joblib")

    researcher = Researcher(query, n_search_queries=n_search_queries, n_sents=n_sents)
    researcher_dict = researcher.to_dict()
    if cache_results:
        dir_path = f"cache/{self.request.id}"
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(researcher_dict, f"{dir_path}/researcher_result_.joblib")
    return researcher_dict

@celery_app.task
def query_to_sentences(researcher_dict, search_query, task_id):
    if load_from_cache:
        return joblib.load(f"{cache_dir}/{search_query}/sentences.joblib")

    researcher = Researcher.from_dict(researcher_dict)
    urls = researcher.get_k_urls(search_query, researcher.results_per_search)
    sentences = researcher.get_content_from_urls(urls, search_query)
    embeddings = embed_sentences(list(s.text for s in sentences))
    relevancies = util.cos_sim(embeddings, researcher.gpt_response_embedding)

    for index, sentence in enumerate(sentences):
        sentence.relevance = relevancies[index].item()
        sentence.embedding = embeddings[index]

    sentence_dicts = [s.to_dict() for s in sentences]
    if cache_results:
        dir_path = f"cache/{task_id}/{search_query}"
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(sentence_dicts, f"{dir_path}/sentences.joblib")
    return sentence_dicts



@celery_app.task
def compress_sentences(sentences):
    embeddings = np.array(list(np.array(s["embedding"]) for s in sentences))
    reduced_data = tsne_model.fit_transform(embeddings)
    return reduced_data.tolist()