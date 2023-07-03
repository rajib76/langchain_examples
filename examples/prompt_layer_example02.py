import os

import promptlayer
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.schema import Document

load_dotenv()
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
prompt_layer_api_key = os.environ.get('prompt_layer_api_key')

openai = promptlayer.openai
openai.api_key = os.environ.get("OPENAI_API_KEY")
promptlayer.api_key = prompt_layer_api_key


class ChatWithPromptLayer():
    def __init__(self):
        self.module = "Chat with Prompt Layer"
        llm = PromptLayerChatOpenAI(pl_tags=["load_qa_chain_run2"], return_pl_id=True, model_name='gpt-3.5-turbo-0613')
        self.llm = llm

    def get_response(self):
        prompt = PromptTemplate(
            template="Answer the user question based on provided context only."
                     "Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question", "context"]
        )

        conversation = load_qa_chain(
            llm=self.llm,
            prompt=prompt,
            chain_type="stuff",
            verbose=True
        )

        return conversation


if __name__ == "__main__":
    cpl = ChatWithPromptLayer()
    convo = cpl.get_response()
    data =[Document(page_content="Langchain is a python based LLM Framework.It also supports java script",metadata={}),
           Document(page_content="Langchain also has integration with many ecosystem tools. For observatory, it integrates with promptlayer",metadata={})]
    while True:
        inpt = input("You: ")
        resp = convo.run(input_documents=data, question=inpt)
        print(resp)
