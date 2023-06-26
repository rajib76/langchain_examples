import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.tools import BaseTool

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
serpapi_api_key = os.environ.get("serapi_key")

llm = OpenAI(temperature=0)


class InformationDiscovery(BaseTool):
    name = "person_information_discovery_tool"
    description = "Useful when we need to discover information about a particular person"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        documents = [Document(page_content="Rajib is a risk averse person. He usually buys low-risk funds",
                              metadata={})]

        return documents

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        raise NotImplementedError("Does not support async")


class FundDiscovery(BaseTool):
    name = "fund_information_discovery_tool"
    description = "Useful when we need to discover information about funds"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        documents = [Document(page_content="Fund ABC is a high risk fund, but CDE is a low risk fund",
                              metadata={})]

        return documents

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        raise NotImplementedError("Does not support async")


tools = [InformationDiscovery(),FundDiscovery()]

model = ChatOpenAI(temperature=0)

planner = load_chat_planner(model)

executor = load_agent_executor(model, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
agent.run("What fund should I recommend to Rajib?")
