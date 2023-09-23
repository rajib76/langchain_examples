import logging
import os
import sys
from typing import Callable, Any, List

import openai.error
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from embeddings.embeddingbase import Embeddings

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeds(Embeddings):
    def __init__(self, engine="text-embedding-ada-002"):
        super().__init__()
        self.module = "Open AI Embedding with retry"
        self.engine = engine

    def _embed_retry_decorator(self,
                               min_seconds=2,
                               max_seconds=5,
                               max_retries=3
                               ) -> Callable[[Any], Any]:

        return retry(
            reraise=True, # want to see the exception encountered at the end of the stack trace
            stop=stop_after_attempt(max_retries), # stop after the max retries, default = 3
            # Exponential backoff with a linear growth, starts with an initial retry attempt
            # and increases the waiting time between retries ensuring that it is never
            # less than 4 secs and never exceeds 10 secs
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                    retry_if_exception_type(openai.error.Timeout)
                    | retry_if_exception_type(openai.error.APIError)
                    | retry_if_exception_type(openai.error.APIConnectionError)
                    | retry_if_exception_type(openai.error.RateLimitError)
                    | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING) # only log failures which will be retried

        )

    def generate_embeddings(self, text: List,
                            min_seconds=2,
                            max_seconds=5,
                            max_retries=3):

        retry_decorator = self._embed_retry_decorator(min_seconds,
                                                      max_seconds,
                                                      max_retries)
        msg = "Started to Embed"
        logger.info(msg)

        @retry_decorator
        def generate_embed_with_retry(text):
            embeddings = openai.Embedding.create(input=text,
                                                 engine=self.engine,
                                                 request_timeout=30)

            # If you want to see the statistics of the retries
            print(generate_embed_with_retry.retry.statistics)

            return embeddings

        try:
            embeddings = generate_embed_with_retry(text)
            prompt_tokens = embeddings.usage["prompt_tokens"]
            total_tokens = embeddings.usage["total_tokens"]
            total_cost = float(total_tokens * (0.0004 / 1000))
            embeds = []
            for i in range(len(embeddings)):
                embeddings = embeddings.data[i].embedding
                print(prompt_tokens, total_tokens, total_cost)
                embeds.append(embeddings)
                print(embeddings)
                return embeds
        except Exception as e:
            print(e)


if __name__ == "__main__":
    emb = OpenAIEmbeds()
    embeds = emb.generate_embeddings(["Hello This is Langchain", "Hi This is Rajib"])
    print(embeds)
