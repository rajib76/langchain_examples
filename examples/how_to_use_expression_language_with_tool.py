import os
from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import BaseTool
from pydantic import Field

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

model = ChatOpenAI(model_name="gpt-3.5-turbo",
                   openai_api_key=OPENAI_API_KEY,
                   temperature=0,
                   max_tokens=1000)


class PoliticalModeration():

    def moderate(self, response):
        if "Indira Gandhi".lower() in response.lower():
            return "It is political and hence moderating"
        else:
            return response


class ReligionModeration():

    def moderate(self, response):
        if "Religion".lower() in response.lower():
            return "It is religious and hence moderating"
        else:
            return response


class ReligiousModerator(BaseTool):
    """Tool that moderates any religious response."""

    name = "moderator"
    description = (
        "A moderator to moderate the response.Useful for when you need to moderate "
        "response from language models."
    )
    moderator_wrapper: ReligionModeration = Field(
        default_factory=ReligionModeration
    )

    def _run(
            self,
            response: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.moderator_wrapper.moderate(response)

    async def _arun(
            self,
            response: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Moderator does not support async")


class PoliticalModerator(BaseTool):
    """Tool that moderates any political response."""

    name = "moderator"
    description = (
        "A moderator to moderate the response.Useful for when you need to moderate "
        "response from language models."
    )
    moderator_wrapper: PoliticalModeration = Field(
        default_factory=PoliticalModeration
    )

    def _run(
            self,
            response: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.moderator_wrapper.moderate(response)

    async def _arun(
            self,
            response: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Moderator does not support async")


response_template = """Use provided tool to moderate the response:

{response}"""
response_prompt = ChatPromptTemplate.from_template(response_template)

query_template = """Answer the below question:

{question}"""
query_template = ChatPromptTemplate.from_template(query_template)

p_moderator = PoliticalModerator()
r_moderator = ReligiousModerator()

chain = query_template | {"response": model} | response_prompt | model | StrOutputParser() | p_moderator | r_moderator
resp = chain.invoke({"question": "What is Python?"})

print(resp)
