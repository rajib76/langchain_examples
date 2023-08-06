import os
from typing import Any

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

#search = SerpAPIWrapper()


class CustomSearchTool(BaseModel):
    name = "Search"
    description = "useful for when you need to answer questions about current events"

    def run(self, query: str, **kwargs: Any) -> str:
        for key in kwargs:
            if key == "context":
                return kwargs["context"]
            else:
                return "No context found"

    async def arun(
            self,
            query: str,
            **kwargs: Any
    ) -> str:
        return "Langchain is a python based LLM farmework"


mysearch = CustomSearchTool()
tools = [
    Tool(
        name="Search",
        func=lambda input : mysearch.run(input,context="Langchain is a python based framework"),
        description="useful for when you need to answer questions about current events"
    )
]

agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

resp = agent.run({"input":"what is Langchain"})

print(resp)
