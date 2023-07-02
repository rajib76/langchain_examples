# export HNSWLIB_NO_NATIVE=1
import os

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.vectorstores import Chroma

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class ChromaExample():
    def __init__(self):
        self.module = "Chroma Example"

    def persist_info(self,embedding_function):
        # save to disk
        docs=[Document(page_content="Langchain is a python based LLM framework",metadata={"source":"internet"})]
        db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
        db2.persist()

if __name__=="__main__":
    cdbex = ChromaExample()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                                  openai_api_key=OPENAI_API_KEY)
    # cdbex.persist_info(embedding_function=embeddings)
    vectordb = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer",return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,
                                                  return_source_documents=True)
    query = "What is langchain"
    llm_response = chain(query)
    print(llm_response)
