import os
from typing import List

import openai
from dotenv import load_dotenv

from embeddings.embeddingbase import Embeddings

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeds(Embeddings):
    def __init__(self):
        super().__init__()
        self.module = "Open AI Embedding"

    def generate_embeddings(self, text: List):
        try:
            embedding = openai.Embedding.create(input=text,
                                                engine="text-embedding-ada-002",
                                                request_timeout=30)

            prompt_tokens = embedding.usage["prompt_tokens"]
            total_tokens = embedding.usage["total_tokens"]
            total_cost = float(total_tokens * (0.0004 / 1000))
            embeddings = embedding.data[0].embedding
            # embedded_text = embedding.embed_documents(text)[0]
            #     print(embedded_text)
            #     total_tokens = cb.total_tokens
            #     prompt_tokens = cb.prompt_tokens
            #     completion_tokens = cb.completion_tokens
            #     print(cb)
            print(prompt_tokens, total_tokens, total_cost)
            print(embeddings)
            return embeddings
        except Exception as e:
            print(e)


if __name__ == "__main__":
    emb = OpenAIEmbeds()
    emb.generate_embeddings(["Hello This is Langchain"])
