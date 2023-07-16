import os
from typing import Optional, List, Any, Union, Dict

from dotenv import load_dotenv
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.input import print_text
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document, AgentFinish, AgentAction, LLMResult
from langchain.tools import BaseTool

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class StdOutCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        print(color)
        self.color = color

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        print("prompt is:\n")
        print(prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> Finished chain.\033[0m")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        print_text(action.log, color=color or self.color)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color or self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        print_text(text, color=color or self.color, end=end)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        print_text(finish.log, color=color or self.color, end="\n")


class ContextRetrievalTool(BaseTool):
    name = "Langchain information retrieval tool"
    description = "Useful when we need to answer question related to langchain"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        documents = [Document(page_content="Langchain is the best LLM framework. It is written in Python and JS",
                              metadata={})]

        return documents

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        raise NotImplementedError("Does not support async")


zero_shot_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=256, verbose=True
)

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}"""


# zero_shot_agent_kwargs = {
#     "prefix": PREFIX,
#     "format_instructions": FORMAT_INSTRUCTIONS,
#     "suffix": SUFFIX,
#     "input_variables": ["input", "chat_history", "agent_scratchpad"],
#     "verbose": True,
# }
zero_shot_agent_kwargs = {}
tools_for_agent=[ContextRetrievalTool()]
zero_shot_agent_obj = ZeroShotAgent.from_llm_and_tools(
    llm=zero_shot_llm, tools=tools_for_agent, **zero_shot_agent_kwargs
)
# print(ZeroShotAgent.create_prompt(
#             tools_for_agent,
#             prefix=PREFIX,
#             suffix=SUFFIX,
#             format_instructions=FORMAT_INSTRUCTIONS,
#             input_variables=["input", "chat_history", "agent_scratchpad"],
#         ))
handler = StdOutCallbackHandler()

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

agent = AgentExecutor.from_agent_and_tools(
    agent=zero_shot_agent_obj,
    tools=tools_for_agent,
    max_iterations=20,
    handle_parsing_errors="Check the output and make sure it conforms to HTML format",
    verbose=True,
    memory=memory
)
# print(ZeroShotAgent.return_values)

agent.run(input="What is langchain", verbose=True,callbacks=[handler])