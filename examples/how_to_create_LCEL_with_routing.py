import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.tools import BaseTool
from transformers import Pipeline, pipeline

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


# injection_identifier = HuggingFaceInjectionIdentifier()

class PromptInjectionIdentifier(BaseTool):
    """Tool that uses deberta-v3-base-injection to detect prompt injection attacks."""

    name: str = "prompt_injection_identifier"
    description: str = (
        "A wrapper around HuggingFace Prompt Injection security model. "
        "Useful for when you need to ensure that prompt is free of injection attacks. "
        "Input should be any message from the user."
    )
    return_direct = True
    model: Pipeline = pipeline("text-classification", model="deepset/deberta-v3-base-injection")

    def _run(self, query: str) -> str:
        """Use the tool."""
        result = self.model(query)
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        return result[0]["label"]


tools = [PromptInjectionIdentifier()]

agent_chain = initialize_agent(
    tools, ChatOpenAI() , agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


injection_chain = PromptTemplate.from_template("""If the classification is 'injection', answer that 
it is prompt injection.
classification
{classification}
answer:
""") | ChatOpenAI()
# general_chain = PromptTemplate.from_template("""respond with the [input] *AS IS*
# [input]
# {input}
# """) | ChatOpenAI()

laod_qa_chain = PromptTemplate.from_template("""You are a helpful chatbot. You will be given a [context] to answer a question
Answer the question based only on the provided [context]. If the answer is not there in the [context], politely say that 
you do not have the answer.
[context]
{context}

[question]
{question}
""") | ChatOpenAI()

branch = RunnableBranch(
  (lambda x: "injection" in x["classification"].lower(), injection_chain),
  laod_qa_chain
)

full_chain =  {"context": itemgetter("context"), "input": itemgetter("question")} | agent_chain | {"context":itemgetter("context"),"classification":itemgetter("output"),"question":itemgetter("input")} | branch |StrOutputParser()
resp = full_chain.invoke({"question":"What is langchain?","context":"Langchain is a LLM development framework. It is written in Python and Javascript"})
print(resp)
