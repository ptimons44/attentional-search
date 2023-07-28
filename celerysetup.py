# celery_setup.py

from flask import Flask
from celery import Celery

from celery import shared_task
from celery.utils.log import get_task_logger


flask_app = Flask(__name__)

celery_app = Celery(flask_app.import_name)
celery_app.config_from_object('celeryconfig')  # Load configurations from celeryconfig.py

logger = get_task_logger(__name__)
