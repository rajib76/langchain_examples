import os

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

loader = WebBaseLoader("https://www.linkedin.com/jobs/view/3655161334/")

data = loader.load()

print(data)

llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo-16k',
    openai_api_key=OPENAI_API_KEY
)

prompt = PromptTemplate(
            template="Answer the user question based on provided context only."
                     "\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question","context"]
        )


conversation = load_qa_chain(
    llm=llm,
    prompt=prompt,
    chain_type="stuff",
    verbose=True
)

# query="What are the skills required for the job"
query="Based on the provided context, what are the technical fraemwork and programming skills required for the job"
res = conversation.run(input_documents=data, question=query)

print(res)

