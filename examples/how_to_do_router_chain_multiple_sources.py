import ast
import os
from operator import itemgetter
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain.tools import BaseTool
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(model_name="gpt-4")


# Creating a tool that will be invoked by the router chain if it cannot answer the question based on the provided
# context It will use the tool to get the information from the wikipedia.
class BuddhistInformation(BaseTool):
    name = "BudhistInformationRetrieval"
    description = "useful for when you need to answer on Buddhism"

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Retrieveing from wikipedia. Buddhism is hard coded, but this can be further genreralized"""
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        content = wikipedia.run("Buddhism")
        return content

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Tool does not support async")


# Creating the tool and making it available to the agent
tools = [BuddhistInformation()]
buddhism_agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# This LLM chain will be used to answer the question if it is available in the context
initial_chain = PromptTemplate.from_template("""Answer based on the below context:
{context}

Question: {question}
"""
                                             ) | OpenAI(model_name="gpt-3.5-turbo-instruct")

# This is the router which will check the answer provided by the context verifier LLM.
# Context Verifier LLM will answer YES or NO based on whether the question can be answered
# based on context.
# If yes, execute the initial_chain
# else call the agent chain
branch = RunnableBranch(
    (lambda x: "NO" in x["answer"], buddhism_agent),
    initial_chain
)

# This the prompt  for the context verifier chain
context_verifier_chain = PromptTemplate.from_template("""Determine if the question can be correctly answered based on the context. Think step by step to answer.
Answer only in a single syllable "NO" or "YES". Please also include the context and the question *AS IS* in the output. 
Output response in a correctly formatted JSON template.

here is an example of the output:
{{"answer":response from the question,"input":the original question,"question":the original question,"context":provided context}}

Context: {context}
Question: {question}
answer:
""") | OpenAI(model_name="gpt-3.5-turbo-instruct")

# This is the final chain which will provide the final answer
final_chain = {"context": itemgetter("context"),
               "question": RunnablePassthrough()} \
              | context_verifier_chain \
              | (StrOutputParser()) \
              | RunnableLambda(ast.literal_eval) \
              | branch

resp = final_chain.invoke(
    {"context": "There are more than 400 million followers in Hinduism", "question": "how many followers do we have "
                                                                                     "in buddhism?"})

print(resp)
