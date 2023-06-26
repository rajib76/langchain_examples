# This program shows how you can use few shot prompting to do guided question and answering
# This program can be further optimized which I will do incrementally
# https://github.com/rajib76/langchain_examples
import os

from dotenv import load_dotenv
from langchain import OpenAI, FewShotPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from llm_prompt_template.guided_question_prompt import GuidedTemplate

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class GuidedQuery():
    def __init__(self):
        self.module = "Guided Query"

    def get_response(self, query, memory):
        # Adding the context. In reality, you will retrieve it from a vector store
        docs = [Document(page_content="To add money in your savings account, you will need to talk to a rep",
                         metadata={}),
                Document(page_content="To add money in your current account, you will need to go to web",
                         metadata={}),
                Document(
                    page_content="To change billing address, you will need to provide your electricity bill as address proof",
                    metadata={}),
                Document(page_content="To change home address, you will need to call the service center",
                         metadata={}),
                Document(page_content="To change your maiden name, you will need to go to web",
                         metadata={}),
                Document(page_content="To change your full name, you will need to call the rep",
                         metadata={})
                ]

        llm = OpenAI(model_name="text-davinci-003",
                     openai_api_key=OPENAI_API_KEY,
                     temperature=0,
                     max_tokens=1000)

        # Adding the few shot examples. I have hard coded this within the code
        # For production grade, you will externalize this. I am just showing
        # the capability
        examples = [
            {
                "question": "How do i change my name?",
                "context": "Changing maiden name requires to go to web. For full name change, talk to a rep",
                "chat_history": "",
                "answer":
                    """
                    Are follow up questions needed here: Yes.
                    Follow up: Which name are you asking, maiden or full name?
                    Human response: I am asking for maiden name
                    Final answer: You can change your maiden name by going to web
                    """
            },
            {"question": "How do i travel to India?",
             "context": "To travel to india you can take a flight or a ship. Flight tickets are expensive but you can reach faster, Travel by ship is cheap but it will take longer time to reach",
             "chat_history": "",
             "answer":
                 """
                    Are follow up questions needed here: Yes.
                    Follow up: Which is important for you? To reach faster or to use a cheap option
                    Human response: cheap option
                    Final answer: You can take a ship to reach India. It will be cheaper than air travel but will take longer time to reach India
                    """
             },
            {
                "question": "How do i change my maiden name?",
                "context": "Changing maiden name requires to go to web. For full name change, talk to a rep",
                "chat_history": "",
                "answer":
                    """
                    Are follow up questions needed here: No.
                    Final answer: You can change maiden name by going to the web
                    """
            },
            {
                "question": "How do i change address in my account?",
                "context": "You can change address of your current account by calling a rep. For savings bank account please go to web",
                "chat_history": "",
                "answer":
                    """
                    Are follow up questions needed here: Yes.
                    Follow up: For which account do you need to change the address?
                    """
            },
            {
                "question": "How do i change address in my current account?",
                "context": "You can change address of your current account by calling a rep. For savings bank account please go to web",
                "chat_history": "",
                "answer":
                    """
                    Are follow up questions needed here: No.
                    Final answer: You can change address of your current account by calling a rep
                    """
            }
        ]
        prompt_template = GuidedTemplate()

        prompt = prompt_template.get_prompt()

        fewshot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=prompt,
            suffix="Question: {input} Context: {context}, {chat_history}",
            input_variables=["context", "input", "chat_history"]
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=fewshot_prompt)
        resp = chain.run(input_documents=docs, input=query, chat_history=memory)
        final_resp = ""
        if "Final answer" in resp:
            final_resp = resp.split("Final answer:")[1]
            return final_resp
        elif "Follow up" in resp:
            final_resp = resp.split("Follow up:")[1]
            return final_resp

if __name__ == "__main__":
    gq = GuidedQuery()
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    while True:
        query = input("You:\n")
        resp = gq.get_response(query, memory)
        print(resp)
