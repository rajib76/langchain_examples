import json
import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

context = "Rahul Gandhi is the prime minister of India"
original_question = "Who is the prime minister of India? "

# Creating the base response chain. We will validate the response from this chain
baseline_response_chain = PromptTemplate.from_template("""You are a helpful chat assistant. You will be given a context and a question.
Please answer in details based on the *CONTEXT* only.If the answer is not there in the context, please do not answer

<context>
{context}
<question>
{question}
Answer:""") | OpenAI(model_name="text-davinci-003") | StrOutputParser()

base_response = baseline_response_chain.invoke({"question": original_question, "context": context})
print("1. Drafts an initial response: \n")
print("----------------------")
print(base_response)
print("----------------------")

# Creating the plan verification chain to generate the verification questions
plan_verifications_chain = PromptTemplate.from_template("""
Given the below question and answer, please generate a list of verification questions that can test 
the accuracy of the original baseline response. 

Here is an example:
Original response: The Indian struggle for freedom started in 1847 with the onset of Sepoy mutiny
Verification question: When did the Indian struggle for freedom start?

Question: {question}
Context: {context}
base_response: {base_response}

{format_instructions}

""") | ChatOpenAI() | StrOutputParser()

# We will use a Pydantic Output Parser to parse the output as a JSON
class PlanVerificationsOutput(BaseModel):
    query: str = Field(description="The user's query")
    base_response: str = Field(description="The response to the user's query")
    context: str = Field(description="The context based on which the response needs to be created")
    facts_and_verification_questions: dict[str, str] = Field(
        description="Facts (as the dictionary keys) extracted from the response and verification questions related to "
                    "the query (as the dictionary values) "
    )


plan_verifications_output_parser = PydanticOutputParser(
    pydantic_object=PlanVerificationsOutput
)

format_instructions = plan_verifications_output_parser.get_format_instructions()

# print("Formatting instruction for the output from plan verification")
# print("--------------------------------------------------------------")
# print(format_instructions)
# print("------------------------------------")

resp = plan_verifications_chain.invoke({"question": original_question,
                                        "context": context,
                                        "base_response": base_response,
                                        "format_instructions": format_instructions})
resp = json.loads(resp)
print("2. Plans verification questions:")
print("--------------------------")
verification_questions = list(resp["facts_and_verification_questions"].values())
print(verification_questions)
print("--------------------------")

answer_verifications_chain = PromptTemplate.from_template("""
Answer the below question based on the context”

Context: {context}
Question: {question}
Answer:
""") | ChatOpenAI() | StrOutputParser()

verify_results_str = ""
for i in range(len(verification_questions)):
    print("3. Answers the questions independently")
    print("-------------------------------------------")
    print("Question: \n")
    question = verification_questions[i]
    print(question)
    print("Answer: \n")
    answer = answer_verifications_chain.invoke({"question": question, "context": context})
    print(answer)
    verify_results_str += f"Question: {question}\nAnswer: {answer}\n\n"

# print(verify_results_str)

# The final response chain
final_response_chain = PromptTemplate.from_template("""Given the ORIGINAL_QUESTION, CONTEXT and the ORIGINAL_RESPONSE,
revise the ORIGINAL_RESPONSE (if applicable) such that it is consistent with information in VERIFIED_SOURCE.
Only keep consistent information.

<ORIGINAL_QUESTION>
{question}
<cONTEXT>
{context}
<ORIGINAL_RESPONSE>
{base_response}

<VERIFIED_SOURCE>
{verify_results_str}

Final response:
""") | ChatOpenAI() | StrOutputParser()

resp = final_response_chain.invoke(
    {"question": original_question, "base_response": base_response, "context": context,
     "verify_results_str": verify_results_str})

print("4. Generates the final verified response")
print("----------------------------------------------")
print(resp)
