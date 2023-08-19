import os
from typing import Any

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import Tool
from langchain_experimental.plan_and_execute.planners.base import LLMPlanner
from langchain_experimental.plan_and_execute.planners.chat_planner import PlanningOutputParser
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class Person(BaseModel):
    age: float = 0.0
    location: str
    current_salary: float = 0.0
    current_expense: float = 0.0
    current_savings: float = 0.0
    expected_future_income: float = 0.0


class RetirementAdvice(BaseModel):
    name = "Retirement Advice for Customers"
    description = "useful for when you need to find out retirement advice for customers"

    def call_my_model(self, features):
        age = features[0].split(":")
        location = features[1]
        income = features[2]
        expense = features[3]
        savings = features[4]
        future_income = features[5]
        final_response = {"age": age,
                          "location": location,
                          "income": income,
                          "expense": expense,
                          "savings": savings,
                          "future_income": future_income}
        ## call your model now to get the prediction
        prediction = "Invest in templeton fund"
        final_response["advice"] = prediction

        return final_response

    def run(self, query: str, **kwargs: Any) -> [{}]:
        features = []
        for key in kwargs:
            if key == "context":
                input_features = kwargs["context"]
                for feature in input_features:
                    features.append(feature.split(":")[1])

                # return features
            else:
                return "No context found"
        response = self.call_my_model(features)
        return response

    async def arun(
            self,
            query: str,
            **kwargs: Any
    ) -> [{}]:
        def run(self, query: str, **kwargs: Any) -> [{}]:
            return query


SYSTEM_PROMPT = (
    f"""Let's first understand the input and person attributes. 
    Please make the follow-up questions based *ONLY* on the provided person attributes in order.
    
    person attributes :{Person.schema_json(indent=2)}
    
    * DO NOT make * any additional follow-up questions which does not help in filling out person attributes.
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

    inputs = {"input": "How do I plan for RAJIB'S retirement?"}
    plan = planner.plan(inputs)
    print(plan)
    print(plan.steps)
    human_answers = []
    for step in plan.steps:
        human_answer = input(step.value + "\n\n")
        human_answers.append(step.value + ":" + human_answer)

    print(human_answers)

    advice = RetirementAdvice()
    tools = [
        Tool(
            name="Retirement Advice for Customers",
            func=lambda input: advice.run(input, context=human_answers),
            description="useful for when you need to find out retirement advice for customers",
            return_direct=False
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    resp = agent.run({"input": "How do I plan for RAJIB'S retirement?"})

    print(resp)
