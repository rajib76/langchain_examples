import datetime
import os

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from langchain_experimental.plan_and_execute.planners.chat_planner import PlanningOutputParser
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class Author(BaseModel):
    author_first_name: str
    author_last_name: str


class Book(BaseModel):
    ISBN: str
    book_name: str
    book_edition: str
    author: Author


SYSTEM_PROMPT = (
    f"""You are a helpful chatbot for an online bookshop. Let's first understand the need of the book shopper. 
    When I say "I", I am referring to the customer who wants to buy the book.
    Please make the follow-up questions based *ONLY* on the provided book and author details in order.

    book details :{Book.model_json_schema()}
    author details :{Author.model_json_schema()}

    * DO NOT make * any additional follow-up questions which does not help in filling out the book details.
     Please output the follow-up questions starting with the header 'Plan:' 
    "and then followed by a numbered list of follow-up questions.
    """)

def load_chat_planner(
        llm: BaseLanguageModel, system_prompt: str = SYSTEM_PROMPT
) -> LLMPlanner:
    """
    Load a chat planner.

    Args:
        llm: Language model.
        system_prompt: System prompt.

    Returns:
        LLMPlanner
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    return LLMPlanner(
        llm_chain=llm_chain,
        output_parser=PlanningOutputParser(),
        stop=["<END_OF_PLAN>"],
    )


if __name__ == "__main__":
    model = OpenAI(model_name="text-davinci-003",
                   openai_api_key=OPENAI_API_KEY,
                   temperature=0,
                   max_tokens=1000)

    planner = load_chat_planner(llm=model)

    while True:
        query = input("Order your book\n\n")
        # I want to buy a book
        inputs = {"input": query}
        plan = planner.plan(inputs)
        human_answers = []
        for step in plan.steps:
            human_answer = input(step.value + "\n\n")
            human_answers.append(step.value + ":" + human_answer)
        print(human_answers)

