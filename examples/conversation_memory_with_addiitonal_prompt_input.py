import os

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from llm_chains.conversation_chain import ConversationChain

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class MyConversation():
    def __init__(self):
        self.module = "My Conversation"
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.memory = memory

    def get_response(self, query):
        PROMPT_TEMPLATE = """
        {chat_history}
        Human: {question}
        AI:"""

        PROMPT = PromptTemplate(input_variables=["question", "chat_history"], template=PROMPT_TEMPLATE)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                         openai_api_key=OPENAI_API_KEY,
                         temperature=0,
                         max_tokens=1000)

        chat_chain = ConversationChain(
            llm=llm,
            input_key="question",
            # You have to define input variables that will be taken from the normal input and memory sequentially.
            prompt=PROMPT
            , memory=self.memory)

        with get_openai_callback() as cb:
            response = chat_chain.predict(question=query, chat_history=self.memory.chat_memory.messages)
            print(response)

            print("Cost and token usage :{cb}".format(cb=cb))


if __name__ == "__main__":
    mc = MyConversation()
    while True:
        query = input("Ask question:\n")
        mc.get_response(query=query)
