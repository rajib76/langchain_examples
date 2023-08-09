import os

from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors.chain_extract import NoOutputParser
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

query = "What is the capital of India?"
docs = [Document(page_content="Delhi is the capital of India",metadata={"source":"wikipedia"}),
        Document(page_content="Dhaka is the capital of Bangladesh",metadata={"source":"wikipedia"}),
        Document(page_content="Today is a bright morning",metadata={"source":"wikipedia"})]

retriever = FAISS.from_documents(docs, OpenAIEmbeddings()).as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold":0.79})

initial_docs = retriever.get_relevant_documents("What is the capital of India?")

print("---Initial Documents-----")
for doc in initial_docs:
        print(doc)


prompt_template = """Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return {no_output_str}.

Remember, *DO NOT* edit the extracted parts of the context.

> Question: {{question}}
> Context:
>>>
{{context}}
>>>
Extracted relevant parts:"""


output_parser = NoOutputParser()

template = prompt_template.format(no_output_str=output_parser.no_output_str)
COMPRESS_DOC_PROMPT = PromptTemplate(
        input_variables= ["question","context"],
        template=template,
        output_parser=NoOutputParser()
)
print(COMPRESS_DOC_PROMPT)

llm = OpenAI(temperature=0)

compressor = LLMChainExtractor.from_llm(llm,prompt=COMPRESS_DOC_PROMPT)
compress_docs = compressor.compress_documents(documents=initial_docs,query=query)

print("---Compressed Documents-----")
for docs in compress_docs:
        print(docs)

