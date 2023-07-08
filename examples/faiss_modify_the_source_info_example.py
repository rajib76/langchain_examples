#This example shows how you can customize the default source name
import os

from dotenv import load_dotenv
from langchain import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Load the environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Load the documents
loader = TextLoader("../data/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# Need to modify the source document to keep only the doc name
modified_docs = [Document(page_content=doc.page_content,metadata={"source":doc.metadata["source"].split("../data/")[1]})
                 for doc in docs]

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(modified_docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
retriever = db.as_retriever()
relevant_docs = retriever.get_relevant_documents(query)
print(relevant_docs[0].metadata['source'])
