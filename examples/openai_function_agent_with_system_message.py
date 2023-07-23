import os
from typing import Optional, List, Any, Union, Dict

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.input import print_text
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, Document, AgentFinish, AgentAction, LLMResult, HumanMessage
from langchain.tools import BaseTool, format_tool_to_openai_function

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
        print("Result is:")
        print(response)

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

    system_message_template = SystemMessage(
        content="You are an helpful AI agent.You answer based on the tool output."
    )

    memBuffer = ConversationBufferMemory(
        memory_key="memory",
        return_messages=True
    )
    model = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=OPENAI_API_KEY)

    while True:
        chat_agent = initialize_agent(llm=model,
                                      tools=tools,
                                      agent=AgentType.OPENAI_FUNCTIONS,
                                      verbose=True,
                                      agent_kwargs={
                                          "extra_prompt_messages": memBuffer.chat_memory.messages,
                                          "system_message": system_message_template
                                      },
                                      memory=memBuffer)
        query = input("Enter your question\n\n")
        resp = chat_agent.run(input=query, callbacks=[StdOutCallbackHandler()])
        print(resp)

