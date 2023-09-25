# Author: Rajib
# A guided question planner which writes a leave application
import os
from datetime import date

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from langchain_experimental.plan_and_execute.planners.chat_planner import PlanningOutputParser
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class Leave(BaseModel):
    days_on_leave: int = 0
    date_of_leave: date

SYSTEM_PROMPT = (
    f"""Let's first understand the question from the applicant. 
    Please make the follow-up questions based *ONLY* on the provided leave attributes in order.

    person attributes :{Leave.schema_json(indent=2)}

    * DO NOT make * any additional follow-up questions which does not help in filling out leave attributes.
    Please output the follow-up questions starting with the header 'Plan:' 
    "and then followed by a numbered list of follow-up questions. ".
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

    inputs = {"input": "I want to take leave?"}
    plan = planner.plan(inputs)
    print(plan)
    print(plan.steps)
    human_answers = []
    for step in plan.steps:
        human_answer = input(step.value + "\n\n")
        human_answers.append(step.value + ":" + human_answer)

    print(human_answers)

    leave_application_template = """
    You are a helpful leave application writer. Please write a leave application based on the inputs provided below.
    [inputs] 
    number of leave days:{no_of_leave_days}
    date of leave: {date_of_leave}
    
    
    Leave Application:"""

    # prompt = PromptTemplate(
    #     template=leave_application_template,
    #     input_variables=["no_of_leave_days", "date_of_leave"]
    # )

    llm_chain = LLMChain(
        llm=model,
        prompt=PromptTemplate.from_template(leave_application_template)
    )

    no_of_leave_days = human_answers[0].split(":")[1]
    date_of_leave = human_answers[1].split(":")[1]

    leave_application_resp = llm_chain.predict(no_of_leave_days=no_of_leave_days, date_of_leave=date_of_leave)

    print(leave_application_resp)

