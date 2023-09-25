# Author : Rajib Deb
# This example shows how to use model graded evaluation
# Inspiration has been taken from the below sources
# https://arxiv.org/pdf/2212.09251.pdf
# https://github.com/openai/evals/blob/main/evals/registry/modelgraded/closedqa.yaml
import os

from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = OpenAI(
    temperature=0,
    model_name="text-davinci-003",
    # model_name='gpt-3.5-turbo-0613',
    openai_api_key=OPENAI_API_KEY
)

# documents = [Document(page_content="Quantum computing is a rapidly-emerging technology that harnesses the laws of "
#                                    "quantum mechanics to solve problems too complex for classical computers. ",
#                       metadata={}),Document(page_content="Today, IBM Quantum makes real quantum hardware — a tool "
#                                                          "scientists only began to imagine three decades ago — "
#                                                          "available to hundreds of thousands of developers. Our "
#                                                          "engineers deliver ever-more-powerful superconducting "
#                                                          "quantum processors at regular intervals, alongside crucial "
#                                                          "advances in software and quantum-classical orchestration. "
#                                                          "This work drives toward the quantum computing speed and "
#                                                          "capacity necessary to change the world.",metadata={})]

documents = [Document(page_content="Monica Cruise is a renowned actor")]
prompt = PromptTemplate(
            template="Answer the user question based on provided context only"
                     "nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question","context"]
        )

# prompt = PromptTemplate(
#             template="Answer the user question based on provided context. If answer is not there in context, respond based on your knowledge."
#                      "nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
#             input_variables=["question","context"]
#         )
conversation = load_qa_chain(
    llm=llm,
    prompt=prompt,
    chain_type="stuff",
    verbose=True)

# question = "What is quantum computing?"
question = "Who is Tom Cruise?"
response = conversation.run(input_documents=documents, question=question)
print(response)

# response = "Quantum computing is a type of blockchain which chains blocks to store information"

# grading_prompt_template = """
# You are assessing a submitted answer on a given question based on a provided context and criterion. Here is the data:
# [BEGIN DATA]
# *** [answer]:    {answer}
# *** [question]:  {question}
# *** [context]:   {context}
# *** [Criterion]: {criteria}
# ***
# [END DATA]
# Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be
# sure that your conclusion is correct. At the end provide a score between 0 and 2"""

grading_prompt_template = """
You are assessing a submitted answer on a given question based on a provided context and criterion. Here is the data: 
[BEGIN DATA] 
*** [answer]:    {answer} 
*** [question]:  {question} 
*** [context]:   {context}
*** [Criterion]: {criteria} 
*** 
[END DATA] 
Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print "Correct" or "Incorrect" (without quotes or punctuation) on its own line corresponding to the correct answer.
The answer will be correct only if it meets all the criterion.
Reasoning:"""

prompt = PromptTemplate(
            template=grading_prompt_template,
            input_variables=["answer","question","context","criteria"]
        )

answer = response
question = question
context=""
for document in documents:
    context = context + document.page_content +"\n\n"

# criteria = "Give one point per criteria: " \
#            "- Check if the answer has been generated from the context ony?" \
#            "- Check if the answer is in line with the question asked?"

criteria = "Criteria: " \
           "relevance:  Is the answer referring to the provided context completely?" \
           "conciseness:  Is the answer concise and to the point? " \
           "correct: Is the answer correct?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(grading_prompt_template)
)

grading = llm_chain.predict(answer=answer, question=question,context=context,criteria=criteria)
print("-----grading is--------")
print(grading)