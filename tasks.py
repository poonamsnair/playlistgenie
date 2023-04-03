from redis import Redis
from rq import Queue

from app import recommendation_function

q = Queue(connection=Redis())

def async_recommendation(user_email, *args, **kwargs):
    result = recommendation_function(*args, **kwargs)
    send_email(user_email, result)