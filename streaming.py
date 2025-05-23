
import threading
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler

class QueueCallbackHandler(BaseCallbackHandler):

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs):
        self.queue.put(None)


def token_generator(run_chain, q):

    threading.Thread(target=run_chain).start()

    while True:
        token = q.get()
        if token is None:
            break
        yield token
