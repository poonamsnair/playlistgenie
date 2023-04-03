import signal
import os

from redis import Redis
from rq import Worker, Queue, Connection

listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISCLOUD_URL', 'redis://localhost:6379')
conn = Redis.from_url(redis_url)

def shutdown_handler(signal, frame):
    print('Shutting down gracefully...')
    worker.stop()

worker = Worker(map(Queue, listen), connection=conn)

signal.signal(signal.SIGTERM, shutdown_handler)

worker.work()
