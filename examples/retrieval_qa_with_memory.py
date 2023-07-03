import os

import pinecone
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


class BRDValidate():
    def __init__(self):
        self.module = "BRDValidate"
        memory = ConversationBufferMemory(output_key="result")
        self.memory=memory

    def get_response(self, query):
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

        embed = OpenAIEmbeddings()
        vectorstore = Pinecone.from_existing_index(index_name="frd", embedding=embed)
        llm = OpenAI(model_name="text-davinci-003",
                     openai_api_key=OPENAI_API_KEY,
                     temperature=0,
                     max_tokens=1000)

        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2,
                                                                                                      "score_threshold": 0.79})

        # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

        template = """You are a helpful reviewer. You review business requirements against functional requirements.
        You will be given a business requirement which you will need to match with the functional requirement provided in the context.
        Answer the question based only on the context provided. Do not make up your answer.
        Answer in the desired format given below.

        Desired format:
        Business requirement: The business requirement given to compare against functional requirement
        Functional requirement: The content of the functional requirement

        {context}
        {question}
        """

        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff",
                                         retriever=retriever,
                                         memory=self.memory,
                                         chain_type_kwargs={
                                             "prompt": PromptTemplate(
                                                 template=template,
                                                 input_variables=["context", "question"],
                                             )},
                                         return_source_documents=True)
        # response = qa.run(query)
        response = qa({"query": query})
        return response


if __name__ == "__main__":
    brd = BRDValidate()
    while True:
        query = input("You\n")
        resp = brd.get_response(query)
        print(resp["result"])
        docs = resp["source_documents"]
        i = 0
        for doc in docs:
            i = i + 1
            print("--------Sources Used----------------")
            print("Source {i}:\n".format(i=i))
            print('-------------------------')
            print(doc.page_content)
