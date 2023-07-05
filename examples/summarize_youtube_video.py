import os

import PyPDF2
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

from examples.loaders.youtube_loader import YouTubeLoader

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class SummarizeContent():
    def __init__(self, url):
        self.url = url

    def extract_content(self):
        url = self.url
        tgt_dir = "./loaders/youtube/"
        ytb_loader = YouTubeLoader(url=url, tgt_dir=tgt_dir)
        docs = ytb_loader()

        return docs


if __name__=="__main__":
    sc = SummarizeContent(url='https://youtu.be/IkXwfhk544g')
    docs = sc.extract_content()

    llm = OpenAI(model_name="text-davinci-003",
                 openai_api_key=OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=1000)


    chain = load_summarize_chain(llm, chain_type="map_reduce")
    resp = chain.run(docs)

    print(resp)

