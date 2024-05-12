import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

graph = Neo4jGraph(url=NEO4J_URI,
                   username=NEO4J_USERNAME,
                   password=NEO4J_PASSWORD)


def get_the_document_content(concept):
    raw_documents = WikipediaLoader(query=concept).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])

    for document in documents:
        print(document)

    return documents


def do_knowledge_modeling(documents):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_transformer = LLMGraphTransformer(llm=llm)

    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print("---graph document---")
    for document in graph_documents:
        print(document)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )


if __name__=="__main__":
    concept = "Agartala"
    documents = get_the_document_content(concept)
    do_knowledge_modeling(documents)
