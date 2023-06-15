import os
from typing import List

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings

from embeddings.embeddingbase import Embeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeds(Embeddings):
    def __init__(self):
        super().__init__()
        self.module = "Open AI Embedding"

    def generate_embeddings(self, text: List):
        try:
            embedding = OpenAIEmbeddings(model="text-embedding-ada-002",
                                         openai_api_key=OPENAI_API_KEY)

            with get_openai_callback() as cb:
                embedded_text = embedding.embed_documents(text)[0]
                print(embedded_text)
                total_tokens = cb.total_tokens
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens
                print(cb)

            return embedded_text
        except Exception as e:
            print(e)