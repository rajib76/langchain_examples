import os
from typing import List, Optional
from urllib import parse

from dotenv import load_dotenv
from kay.rag.retrievers import KayRetriever
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine

load_dotenv()
KAY_API_KEY = os.environ.get('KAY_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


# Using native kay SDK
# from kay.rag.retrievers import KayRetriever
#
# # Initialize the retriever
# retriever = KayRetriever(dataset_id = "company",  data_types=["10-K", "10-Q", "8-K", "PressRelease"])
#
# # Query the retriever
# context = retriever.query(query="What is the chance of a recession?",num_context=3)
#
# # Examine the retrieved context and then append it to the prompt before you call your LLM
# print(context[0])

# LangChain Way

class SnowflakeDBTool(BaseTool):
    name = "Snowflake DB tool"
    description = "useful for when you need to retrieve company stock data from snowflake"

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        sf_prompt_template = "Given an input question and a *dbt* schema and sources, " \
                             "first create and execute a * syntactically * correct {{dialect}} query to run, " \
                             "then look at the results of the query and return the answer. " \
                             "dbtschema:{schema}\n\n Question: {{input}}."

        sf_user = os.environ.get('sf_user')
        sf_pass = parse.quote(os.environ.get('sf_pass'))
        sf_schema = os.environ.get('sf_schema')
        sf_db = os.environ.get('sf_db')
        sf_warehouse = os.environ.get('sf_warehouse')

        engine = create_engine(
            'snowflake://{sf_user}:{sf_pass}@ju31715.us-east-2.aws/{sf_db}/{sf_schema}?warehouse={sf_warehouse}'.format(
                sf_user=sf_user,
                sf_pass=sf_pass,
                sf_warehouse=sf_warehouse,
                sf_schema=sf_schema,
                sf_db=sf_db
            )
        )

        sql_db_chain = SQLDatabase(engine)
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)

        schema = f"""
        version: 2

        models:
          - name: stocks
            description: This table contains the stock information of companies
            columns:
              - name: stock_name
                description: Name of the stock of the company
              - name: revenue_in_billions
                description: The revenue of the company in billions
              - name: net_income_in_billions
                description: The net income of the company in billions
              - name: diluted_eps
                description: The diluted eps of the company.
              - name: net_profit_margin_percent
                description: The net profit margin of the company."""

        template = sf_prompt_template.format(schema=schema)

        SQL_PROMPT = PromptTemplate(
            input_variables=["input", "dialect"],
            template=template
        )
        dbchain = SQLDatabaseChain.from_llm(llm,
                                            db=sql_db_chain,
                                            prompt=SQL_PROMPT,
                                            verbose=True,
                                            return_intermediate_steps=True)

        dbchain.top_k = 100
        # resp = dbchain.run(input)
        resp = dbchain({"query": query})

        return resp['result']


    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Does not support async")


class InvestmentAdviceTool(BaseTool):
    name = "investment advice"
    description = "useful for when you need to advice your customers on wealth investment"

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List:
        """Use the tool."""
        retriever = KayRetriever(dataset_id="company", data_types=["PressRelease"])
        context = retriever.query(query=query, num_context=3)
        return context

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Does not support async")


# tools = [InvestmentAdviceTool()]
tools = [InvestmentAdviceTool(),SnowflakeDBTool()]
model = ChatOpenAI(model_name="gpt-4")
agent = initialize_agent(
    tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

resp = agent.run("Is NVIDIA Corp a good stock to invest in? Please also share the details of NVIDIA Corp stock.")
# resp = agent.run("Please share the details of NVIDIA Corp stock?")
print(resp)
