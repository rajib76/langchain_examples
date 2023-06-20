import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.agents import ConversationalChatAgent, AgentType, AgentExecutor
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain.schema import Document, ChatMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class ContextRetrievalTool(BaseTool):
    name = "Langchain information retrieval tool"
    description = "Useful when we need to answer question related to langchain"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        documents = [Document(page_content="Langchain is the best LLM framework. It is written in Python and JS",
                              metadata={})]

        return documents

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Document]:
        raise NotImplementedError("Does not support async")


class ConversationReactDesc():
    def __init__(self):
        self.module = "Conversation React"
        self.chat_history = []

    def get_context(self):
        documents = [Document(page_content="Langchain is the best LLM framework. It is written in Python and JS",
                              metadata={})]

        return documents

    def create_chain(self, query):
        messages = [
            SystemMessagePromptTemplate.from_template(
                "You are a helpful chatbot.You have acces to ContextRetrievalTool. "
                "You answer based on only the context returned by the tool. "
                "If the question cannot be answered based on the retrieved context, please do not answer."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                " - Always answer based on the content retrieved by the tool"
                " - You have access to the following tool"
                " - Langchain information retrieval tool:Useful when we need to answer question related to langchain"
                " - If you don't know the answer truthfully say yo don't have an answer. Don't try to make up an answer."
                "   {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        # messages = [
        #     SystemMessagePromptTemplate.from_template(
        #         "You are a helpful chatbot.You have acces to ContextRetrievalTool. "
        #         "You answer based on only the context returned by the tool. "
        #         "If the question cannot be answered based on the retrieved context, please do not answer."),
        #     MessagesPlaceholder(variable_name="chat_history"),
        #     HumanMessagePromptTemplate.from_template(
        #         "Answer question based only on the retrieved context from ContextRetrievalTool /nQuestion: {"
        #         "question}"),
        #     MessagesPlaceholder(variable_name="agent_scratchpad"),
        # ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         openai_api_key=OPENAI_API_KEY,
                         temperature=0,
                         max_tokens=1000)

        tools = [ContextRetrievalTool()]
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm, tools=tools, memory=memory,
            verbose=True, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION)

        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, verbose=True,
            tools=tools)

        self.chat_history.append(HumanMessage(content=query))

        resp = agent_chain.run(
            {
                "input": chat_prompt,
                "chat_history": self.chat_history
            }

        )
        self.chat_history.append(AIMessage(content=resp))

        print(self.chat_history)

        return resp


if __name__ == "__main__":
    crd = ConversationReactDesc()
    while True:
        query = input("What is your question?\n")
        resp = crd.create_chain(query)
        print(resp)
