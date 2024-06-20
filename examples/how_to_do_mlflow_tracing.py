# pip install
# openai==1.30.5
# langchain==0.2.1
# langchain-openai==0.1.8
# langchain-community==0.2.1
# mlflow==2.14.0
# tiktoken==0.7.0

import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

import mlflow

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Ensure that you have started MLFLOW with the below command
# mlflow server --host 127.0.0.1 --port 8080
# now you can point it to the server
mlflow.set_tracking_uri("http://localhost:8080")

# Create a new experiment. This is where the model and the traces will be looged
mlflow.set_experiment("Q&A Tracing")

mlflow.langchain.autolog(log_models=True, log_input_examples=True)

llm = OpenAI(temperature=0.7, max_tokens=1000)

question_prompt = PromptTemplate(
    template ="""
    You are a helpful chat agent. You answer questions based on provided context only. 
    {context}
    {question}
    answer:
    """
)

chain = question_prompt | llm

chain.invoke(
    {
        "context": """The Taj Mahal (/ˌtɑːdʒ məˈhɑːl, ˌtɑːʒ-/; lit. 'Crown of the Palace') is an ivory-white marble 
        mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by 
        the fifth Mughal emperor, Shah Jahan (r. 1628–1658) to house the tomb of his beloved wife, Mumtaz Mahal; it 
        also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare (42-acre) complex, 
        which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a 
        crenellated wall.""",
        "question": "Where is TajMahal situated?",
    }
)
