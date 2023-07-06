import os

from dotenv import load_dotenv
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

from examples.plan_and_execute_agent_v1 import search, llm_math_chain

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


llm = ChatOpenAI(temperature=0)

tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
),
    Tool(
        name="Calculator",
        func = llm_math_chain.run,
        description="useful for arithmetic. Expects strict numeric input, no words.",
    ),
]
agent = ConversationalChatAgent.from_llm_and_tools( llm = llm,
                                                    tools = tools,
                                                    system_message= "Assistant is useful for general purposes and will try its best to answer questions",
                                                    input_variables=['input','chat_history', 'agent_scratchpad'])


agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ),
    )

resp = agent_executor.run(input="When Indira Gandhi died, How old was Rajiv Gandhi?")

print(resp)
