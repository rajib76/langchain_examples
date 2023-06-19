import os

import PyPDF2
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document


# https://community.cisco.com/kxiwq67737/attachments/kxiwq67737/5926-discussions-contact-center/110352/1/Router%20Error%20Codes%20-%20DocWiki.pdf
# The document Router_Error_Codes.pdf has been copied from the above link and is a PDF file.

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class SummarizeContent():
    def __init__(self, document_location):
        self.document_location = document_location

    def extract_content(self):
        page_content = []
        reader = PyPDF2.PdfReader(self.document_location)
        for page in reader.pages:
            page_content.append(page.extract_text())
        docs = [Document(page_content=page) for page in page_content]
        # for doc in docs:
        #     print(doc.page_content)
        #     print(doc.metadata)
        return docs


if __name__=="__main__":
    sc = SummarizeContent(document_location='../data/Router_Error_Codes.pdf')
    docs = sc.extract_content()

    llm = OpenAI(model_name="text-davinci-003",
                 openai_api_key=OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=1000)


    chain = load_summarize_chain(llm, chain_type="map_reduce")
    resp = chain.run(docs)

    print(resp)

