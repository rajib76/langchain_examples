import os

import nltk
from langchain.document_loaders import UnstructuredPowerPointLoader
import nltk
import ssl
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def download_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download()


# download_nltk()

def get_ppt_data(path):
    loader = UnstructuredPowerPointLoader(path)
    data = loader.load()

    return data


if __name__ == "__main__":
    # download_nltk() # Run first time to download nltk
    path = "/Users/joyeed/Documents/TEST_PPT.pptx"
    documents = get_ppt_data(path)
    for document in documents:
        print(document.page_content)
        print(document.metadata)
