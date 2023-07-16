import os

from dotenv import load_dotenv
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
serpapi_api_key = os.environ.get('serapi_key')


def get_llm_instance():
    llm = OpenAI(model="text-davinci-003", temperature=0)
    return llm


def get_list_of_tools():
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to search internet to provide an answer",
        )
    ]

    return tools


if __name__ == "__main__":
    llm = get_llm_instance()
    tools = get_list_of_tools()

    self_ask_with_search = initialize_agent(
        tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)

    response = self_ask_with_search.run(
        "Who won the Wimbledon 2023?"
    )

    print("The answer is: /n")
    print(response)
