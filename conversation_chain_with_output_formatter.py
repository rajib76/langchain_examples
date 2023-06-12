import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory, ConversationStringBufferMemory
from langchain.schema import Document

from llm_chains.conversation_chain import ConversationChain
from llm_prompt_template.example_template_chain import ExampleTemplateChain

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class MyConversation():
    def __init__(self):
        self.module = "My Conversation"

    def get_response(self, query, memory=None):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     openai_api_key=OPENAI_API_KEY,
                     temperature=0,
                     max_tokens=1000)

        docs = [
            Document(page_content="Langchain is a python based framework. I love it. It makes my gen ai coding easier",
                     metadata={})]

        et = ExampleTemplateChain()
        prompt, out_parser = et.get_prompt()
        _input = prompt.format_prompt(question=query, context=docs)

        print(_input)

        chat_chain = ConversationChain(
            llm=llm,
            verbose=False,
            memory=memory

        )

        with get_openai_callback() as cb:
            response = chat_chain.predict(input=_input.to_string())

            print(response)
            resp_json = out_parser.parse(response)

            print("Cost and token usage :{cb}".format(cb=cb))

            return resp_json


if __name__ == "__main__":
    my_conversation = MyConversation()
    # my_conversation_history = ConversationBufferMemory()
    my_conversation_history = ConversationStringBufferMemory()
    while True:
        question = input("Enter your question\n\n")
        resp_json = my_conversation.get_response(question, memory=my_conversation_history)
        print("Answer : {answer}".format(answer = resp_json["answer"]))
        print("\n")
        print("Source : {source}".format(source=resp_json["source"]))
