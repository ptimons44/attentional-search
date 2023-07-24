from celerysetup import celery_app
from query_graph.pipeline import get_llm_response, get_web_content

from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)

@celery_app.task(bind=True)
def get_llm_response_task(self, query, n_sents=150):
    logger.info("Getting LLM response task started.")
    result = get_llm_response(query, n_sents)
    logger.info("Getting LLM response task completed!")
    return result

@celery_app.task(bind=True)
def get_web_content_task(self, researcher_dict, n_sents=150):
    logger.info("Getting web content task started.")
    result = get_web_content(researcher_dict, n_sents)
    logger.info("Getting web content task completed!")
    return result

