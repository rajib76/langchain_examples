import os

from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

docs = [Document(page_content="main program:abc\ndata division:01 x comp-1 value 1.\n88 done value 0.\n01 y comp-1.\n01 z comp-1.\n01 mul_or_div pic x(8).",
                 metadata={}),
        Document(page_content="main program:abc\nprocedure division:perform until done\ndisplay 'Enter first number (enter 0 to end the program):\naccept value of x\nif not done\ndisplay 'Enter second number (not 0 if dividing): '\naccept value of y\ndisplay 'Enter mul or div: '\naccept value of mul_or_div \nif mul_or_div equals 'mul' \ncall 'mul' using x y z \ndisplay value 'Product is: ' z \nelse \ncall 'div' using x y z \ndisplay value 'Quotient is: ' z.\nexit program.",
                 metadata={}),
        Document(
            page_content="sub program:div\nlinkage section:01 a comp-1.\n01 b comp-1.\n01 c comp-1.\nprocedure division using a b c.\ncompute c = a/b.\nexit program.",
            metadata={}),
        ]

template = """You are a helpful cobol programmer. You will understand the logic of cobol programs 
and help identify enhancements that are required withing the program and the subprograms
based on the code snippet provided as context. 
Answer the question based only on the context provided. Do not make up your answer.
Answer in the desired format given below.

Desired format:
Program Name: The name of the program which requires change
Code snippet: The piece of code that requires a change

{context}
{question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question", "context"]
)

llm = OpenAI(model_name="gpt-4",
                     openai_api_key=OPENAI_API_KEY,
                     temperature=0,
                     max_tokens=1000)

query = "I would like to change the current div sub program to be able to handle zero divide error. how do I change the programs"
chain = load_qa_chain(llm, chain_type="stuff",prompt=prompt)
resp = chain.run(input_documents=docs, question=query)

print(resp)