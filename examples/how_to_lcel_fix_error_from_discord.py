import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

loader = DirectoryLoader("/Users/joyeed/langchain_examples/langchain_examples/data/", glob='**/*.md')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text = text_splitter.split_documents(documents)
print(text)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# vectorstore = FAISS.from_texts(
#     ["twitter:baseball is a great sport","twitter:baseball is played in a field"], embedding=OpenAIEmbeddings()
# )
vectorstore = FAISS.from_documents(text, embeddings)

retriever = vectorstore.as_retriever()

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the below question based on provided context.
    REMEMBER, you need to answer based on the context only.
    
    {context}
    
    Question: {question}
    """
)

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Compose the chain for generating posts

question = "Write 2 {platform} posts about {topic}".format(platform="twitter",topic="baseball")
chain = (
    {"context": retriever,"question": RunnablePassthrough()}
    | prompt_template
    | model
    | StrOutputParser()
)

# Invoke the chain to generate a post
output = chain.invoke(question)

# Print the generated post
print(output)
