import os

from dotenv import load_dotenv
from langchain import OpenAI, LLMChain
from langchain.agents import create_csv_agent, AgentType
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

chain1 = create_csv_agent(
    OpenAI(temperature=0, openai_api_key = OPENAI_API_KEY),
    # employee_id , employee_name
    # 1, Rajib
    # 2, Mohan
    "../data/employee.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

llm = OpenAI(temperature=.7)
template = """Given the name of the employee and the below context, find the department of the emplyree.
Please answer based on the provided context only. If the department in not there in the context,respond with 'NO DEPT'.

context:
{context}
Employee:
{output}
Department:"""
prompt_template = PromptTemplate(input_variables=["output","context"], template=template)
chain2 = LLMChain(llm=llm, prompt=prompt_template, output_key="department")

context = "rajib is from finance department and mohan is from accounting."

overall_chain = SequentialChain(chains =[chain1,chain2],
                                      input_variables=["input","context"],
                                      output_variables=["output","department"],
                                      verbose=True)

resp = overall_chain({"input":"What is the name and department of employee with id 1 ?","context":context})
print(resp["department"])