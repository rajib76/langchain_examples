import os

import requests
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI


class MyStreamHandler(BaseCallbackHandler):
    def __init__(self, url: str, start_token=""):
        self.url = url
        self.text = start_token

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token)
        self.text += token
        data = self.text
        r = requests.post("http://127.0.0.1:8000/generate", data=data)


load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def get_response(query: str):
    stream_handler = MyStreamHandler(url="http://127.0.0.1:8000",start_token="Crafting response...")
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-0613',
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler]

    )

    prompt_template = PromptTemplate(
        template="Answer the user question based on your knowledge"
                 "{question}\n\nAnswer:",
        input_variables=["question"]
    )

    llm_chain = LLMChain(llm=llm,
                         prompt=prompt_template)

    response = llm_chain(query)
    return response


if __name__ == "__main__":
    get_response("Summarize the history of Indian Independence struggle from british in 50 sentences")
