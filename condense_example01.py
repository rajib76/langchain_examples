import os

from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
{context}
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = OpenAI(model_name="text-davinci-003",
                 openai_api_key=OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=1000)

memory = ConversationBufferMemory(memory_key="chat_history",input_key="question")
condense_chain = load_qa_chain(llm,chain_type="stuff",memory=memory,prompt=CONDENSE_QUESTION_PROMPT)
docs = [Document(page_content="Hi I am from California. I like to eat icecream",metadata={})]
while True:
    query = input("Ask a question: \n")
    response = condense_chain({
        "input_documents": docs,
        "question": query
    },return_only_outputs=True)

    print(response)
