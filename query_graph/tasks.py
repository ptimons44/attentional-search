from query_graph.researcher import Researcher

from celerysetup import celery_app

@celery_app.task
def init_researcher(query, n_sents=100):
    researcher = Researcher(query, n_sents=n_sents)
    return researcher.to_dict()

@celery_app.task
def probe_search_query(researcher_dict, search_query):
    researcher = Researcher.from_dict(researcher_dict)
    urls = researcher.get_k_urls(search_query, researcher.results_per_search)
    sentences = [sentence for sentence in researcher.get_content_from_url(url) for url in urls]
    researcher.sentences = sentences
    return researcher.to_dict()

@celery_app.task
def get_top_n_sents(researcher_dict):
    researcher = Researcher.from_dict(researcher_dict)
    sentences = researcher.get_top_n_sents()
    return sentences

@celery_app.task
def debug_task():
    print('Debug task!')
    return 'Debug task!'

# from transformers import BertTokenizer, BertModel
# def bert_tokenizer_and_model():
#     model_name = 'bert-base-uncased'
#     model = BertModel.from_pretrained(model_name, output_attentions=True)
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     return tokenizer, model