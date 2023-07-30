import json
import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
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
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    print(functions)
    model = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=OPENAI_API_KEY)
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
    }
    chat_agent = initialize_agent(llm=model, tools=tools, memory=memory,agent=AgentType.OPENAI_FUNCTIONS, verbose=True,
                                  agent_kwargs=agent_kwargs)

    while True:
        query = input("Enter your question\n")
        resp = chat_agent.run(query)
        print(resp)
