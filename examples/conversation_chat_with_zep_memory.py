# Source of this code :https://python.langchain.com/docs/modules/data_connection/retrievers/integrations/zep_memorystore.html
from langchain.memory.chat_message_histories import ZepChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# Set this to your Zep server URL
ZEP_API_URL = "http://localhost:8000"

session_id = str(uuid4())  # This is a unique identifier for the user

# Set up Zep Chat History
zep_chat_history = ZepChatMessageHistory(
    session_id=session_id,
    url=ZEP_API_URL,
)

ddg = DuckDuckGoSearchRun()
tools = [ddg]

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=zep_chat_history
)

# Initialize the agent
llm = OpenAI(temperature=0)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

test_history = [
    {"role": "human", "content": "Who was Octavia Butler?"},
    {
        "role": "ai",
        "content": (
            "Octavia Estelle Butler (June 22, 1947 – February 24, 2006) was an American"
            " science fiction author."
        ),
    },
    {"role": "human", "content": "Which books of hers were made into movies?"},
    {
        "role": "ai",
        "content": (
            "The most well-known adaptation of Octavia Butler's work is the FX series"
            " Kindred, based on her novel of the same name."
        ),
    },
    {"role": "human", "content": "Who were her contemporaries?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R."
            " Delany, and Joanna Russ."
        ),
    },
    {"role": "human", "content": "What awards did she win?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur"
            " Fellowship."
        ),
    },
    {
        "role": "human",
        "content": "Which other women sci-fi writers might I want to read?",
    },
    {
        "role": "ai",
        "content": "You might want to read Ursula K. Le Guin or Joanna Russ.",
    },
    {
        "role": "human",
        "content": (
            "Write a short synopsis of Butler's book, Parable of the Sower. What is it"
            " about?"
        ),
    },
    {
        "role": "ai",
        "content": (
            "Parable of the Sower is a science fiction novel by Octavia Butler,"
            " published in 1993. It follows the story of Lauren Olamina, a young woman"
            " living in a dystopian future where society has collapsed due to"
            " environmental disasters, poverty, and violence."
        ),
    },
]

for msg in test_history:
    zep_chat_history.add_message(
        HumanMessage(content=msg["content"])
        if msg["role"] == "human"
        else AIMessage(content=msg["content"])
    )
agent_chain.run(
    input="What was I talking about?"
)