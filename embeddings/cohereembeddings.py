import os
from typing import List

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings

from embeddings.embeddingbase import Embeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("cohere_api_key")


class CohereEmbeds(Embeddings):
    def __init__(self):
        super().__init__()
        self.module = "Cohere Embedding"

    def generate_embeddings(self, text: List):
        try:
            embedding = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=COHERE_API_KEY)

            with get_openai_callback() as cb:
                text=[text]
                embedded_text = embedding.embed_documents(text)[0]
                print(embedded_text)
                total_tokens = cb.total_tokens
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens
                print(cb)

            return embedded_text
        except Exception as e:
            print(e)

if __name__=="__main__":
    ce = CohereEmbeds()
    ce.generate_embeddings("I am boy")