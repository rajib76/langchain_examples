# This code is an example where serpapi is called as a tool
# When the agent is getting the information from the tool, the information
# is also being written in a file (see the below lines) so that it can be used for citation
#         with open("/Users/joyeed/observations/obs"+str(uuid_str)+".txt","w") as f:
#             f.write(thoughts)
import os
import os
import re
import uuid
from typing import Any, Optional, List, Tuple, Union, Callable

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent, LLMSingleActionAgent, AgentOutputParser
from langchain.agents.mrkl.output_parser import FINAL_ANSWER_ACTION
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import Document, AgentAction, AgentFinish, OutputParserException
from pydantic import BaseModel
from serpapi import GoogleSearch

from examples.plan_and_execute_agent_v1 import llm_math_chain

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
serpapi_api_key = os.environ.get('serapi_key')

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Craft the final answer to the original input question based on tool output"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prefix = PREFIX
suffix = SUFFIX
format_instructions = FORMAT_INSTRUCTIONS


class MySerpAPI(BaseModel):

    def run(self, query: str, **kwargs: Any) -> {}:
        print("I am calling serpapi")
        params = {
            "engine": "google",
            "q": query,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com",
            "api_key": serpapi_api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        print(results)
        return {"answer":results["knowledge_graph"],"source":""}
        #return {"answer":results["organic_results"][0]["snippet"], "source":results["organic_results"][0]["link"]}
        # return results["organic_results"][0]["snippet"]
        #return results["organic_results"]
        # resp_json = ast.literal_eval(json.dumps(results))
        # resp = results["answer_box"]
        # ans = resp["answer"]
        # source = resp["link"]
        # return {"answer": ans, "source": source}
        # return resp_json

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        raise NotImplementedError("Does not support async")


search = MySerpAPI()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about past and current events",
        return_direct=False,
    ),
Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]


def get_tools(query):
    ALL_TOOLS = tools
    return ALL_TOOLS


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        uuid_str= uuid.uuid4()
        with open("/Users/joyeed/observations/obs"+str(uuid_str)+".txt","w") as f:
            f.write(thoughts)

        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################

        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        print(self.template.format(**kwargs))
        return self.template.format(**kwargs)

class MRKLOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    "Parsing LLM output produced both a final answer "
                    f"and a parse-able action: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format: Missing 'Action:' after 'Thought:'",
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format:"
                " Missing 'Action Input:' after 'Action:'",
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl"



class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


class MySearchAgent(BaseSingleActionAgent):
    """A custom agent that can search the internet, find answer and cite sources."""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")


tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
# tool_names = ", ".join([tool.name for tool in tools])
# format_instructions = format_instructions.format(tool_names=tool_names)
template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
input_variables = ["input", "intermediate_steps"]

prompt = CustomPromptTemplate(template=template, tools_getter=get_tools, input_variables=input_variables)

print(prompt)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
             openai_api_key=OPENAI_API_KEY,
             temperature=0,
             max_tokens=1000)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
#output_parser = CustomOutputParser()
output_parser = MRKLOutputParser()

# mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)


agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)



# agent = MySearchAgent()

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

resp = agent_executor.run("When did FIFA 2022 happen?")
print(resp)
# resp_json = ast.literal_eval(json.dumps(resp))
#
# print("Answer :", resp_json["answer"])
# print("Source :", resp_json["source"])


# resp_json = ast.literal_eval(json.dumps(resp))
# print("response is:", resp_json)
#
# print("Final answer is ", resp_json['answer'])
# print("Source of answer is ", resp_json['source'])
