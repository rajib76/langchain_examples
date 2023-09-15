# Author : Rajib Deb
# This example shows how to use RetrievalQAChain with
# 1. Chroma DB as vector store
# 2. memory
# 3. Change the default euclidean similarity to cosine
# 4. How to pass the similarity score threshold and # of documents to return
# This example first loads the Chroma db with the PDF content - Execute this only once(see somment below)
# Then it retrieves the relevant documents
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class RetrievalQAChain():
    def __init__(self):
        self.module = "RetrievalQAChain"
        self.embedding_function = OpenAIEmbeddings()

    def load_chroma(self, docs):
        chroma_db = Chroma.from_documents(docs, self.embedding_function, persist_directory="../db/chroma_db")
        chroma_db.persist()

        return chroma_db

    def return_retriever(self):
        chroma_db = Chroma(persist_directory="../db/chroma_db",
                           embedding_function=self.embedding_function,
                           collection_metadata={"hnsw:space": "cosine"})

        return chroma_db


if __name__ == "__main__":
    pdf_loc = "../data/essay/167.full.pdf"
    loader = PyPDFLoader(pdf_loc)
    documents = loader.load()
    r_qa_chain = RetrievalQAChain()
    # Uncomment the below code and run once to load the CHROMADB
    # r_qa_chain.load_chroma(documents)

    chroma_db = r_qa_chain.return_retriever()
    # At the top we initialized the Chroma DB with cosine relevance function
    # here score_threshold is the cosine distance and not the similarity
    # cosine_distane = 1- cosine_similarity
    # use 0.2 which is 1 - 0.79
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                       search_kwargs={'score_threshold': 0.21,'k':5})

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer")
    chain = RetrievalQAWithSourcesChain.from_chain_type(ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0),
                                                        chain_type="stuff",
                                                        retriever=retriever,
                                                        return_source_documents=True,
                                                        memory=memory)

    while True:
        question = input("Enter your question:\n\n ")
        resp = chain({"question": question})
        print(resp)