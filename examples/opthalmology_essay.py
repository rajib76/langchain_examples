import os

import PyPDF2
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

embedding_function = OpenAIEmbeddings()

def query_chroma(query):
    db3 = Chroma(persist_directory="./opthalmology/chroma_db", embedding_function=embedding_function)
    docs = db3.similarity_search(query,k=10)
    return docs
    # print(len(docs))
    # for doc in docs:
    #     print(doc.page_content)

def persist_info(embedding_function, docs):
    # save to disk
    # docs = [Document(page_content="Langchain is a python based LLM framework", metadata={"source": "internet"})]
    db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./opthalmology/chroma_db")
    db2.persist()


def chunk_document():
    reader = PyPDF2.PdfReader('/Users/joyeed/langchain_examples/langchain_examples/data/essay/167.full.pdf')
    text_splitter = TokenTextSplitter(chunk_size=1400, chunk_overlap=200)

    # chunked_embed_texts=[]
    # chunked_texts = []
    docs = []
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=text, metadata={"source": "pdf.pdf"}) for text in texts]

    return docs
    # for text in texts:
    #     chunked_text = embedding.generate_embeddings(texts)
    #     chunked_texts.append(chunked_text)
    #     chunked_embed_texts.append(chunked_text)
    #
    # ziprows = zip(chunked_texts,chunked_embed_texts)
    # print(texts)


if __name__ == "__main__":
    # docs = chunk_document()
    # print(len(docs))
    # persist_info(embedding_function, docs)
    query = "Generate the essay on opthalmology in not more than 1000 words.cite the sources."
    docs = query_chroma(query)
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-0613',
        openai_api_key=OPENAI_API_KEY
    )

    # prompt = PromptTemplate(
    #     template="Answer the user question based on provided context only."
    #              "\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    #     input_variables=["question", "context"]
    # )


    prompt = PromptTemplate(template="Your are a helpful essay writer. "
                                     "You write essays following harvard referencing style."
                                     "You must write the essay using the provided context only."
                                     "The essay must not be biased and should have multiple perspective. "
                                     "It must be factually correct and should be suitable for scientific community."
                                     "\n\n"
                                     "context:{context} "
                                     "Based on the context provided {question}",
                            input_variables=["question", "context"])


    conversation = load_qa_chain(
        llm=llm,
        prompt=prompt,
        chain_type="stuff",
        verbose=True
    )

    res = conversation.run(input_documents=docs, question=query)
    print(res)


# documents = []
# directory = '/Users/joyeed/langchain_examples/langchain_examples/data/essay/'
# for filename in os.listdir(directory):
#     if filename.endswith('.pdf'):
#         print(filename)
#         loaders = (UnstructuredPDFLoader(directory+"/"+filename))
#         documents.extend(loaders.load())
#
# print("The total number of documents is " + str(len(documents)))
#
