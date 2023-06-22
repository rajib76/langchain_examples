import json
import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, Document
from langchain.tools import Tool, BaseTool, format_tool_to_openai_function

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class ContextRetrievalTool(BaseTool):
    name = "langchain_information_retrieval_tool"
    description = "Useful when we need to answer question related to langchain"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        documents = [Document(page_content="Langchain is the best LLM framework. It is written in Python and JS",
                              metadata={})]

        return documents

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        raise NotImplementedError("Does not support async")


if __name__ == "__main__":
    tools = [ContextRetrievalTool()]
    functions = [format_tool_to_openai_function(t) for t in tools]

    print(functions)
    model = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=OPENAI_API_KEY)
    chat_agent = initialize_agent(llm=model, tools=tools, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    resp = chat_agent.run("What is Langchain")
    print(resp)
