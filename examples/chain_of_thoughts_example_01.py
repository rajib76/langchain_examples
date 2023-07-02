# Example to show chain of thought
import os

from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Prefix of the prompt
prefix = "You are a helpful chatbot and answer questions based on provided context only. If the answer to the question is not there in the context, you can politely say that you do not have the answer"

# Examples of chain of thought to be included in the prompt
EXAMPLES=["""Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be based on {context}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""]

# Template to be used
example_template ="""
Context: {context}
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables = ["context","query","answer"],
    template = example_template
)

suffix = """
Context: {context}
User: {query}
AI:
"""

CHAT_PROMPT = PromptTemplate.from_examples(
    examples = EXAMPLES, suffix=suffix, input_variables=["context","query"],prefix=prefix
)

# query = "I want to buy stocks of Google. Can I buy through your bank"
# context = "Bank customers will not be able to trade in shares and mutual funds through their bank account. " \
#           "They will need to open a trading account for trading in the market"

#print(CHAT_PROMPT.format(query=query,context=context))

#query = "I want to buy stocks of Google. Can I buy through your bank"
#query = "Is the bank open on 25th December"
# context = "Langchain in a python based llm framework. It was created in 2023 by Harrison chase"
#print(TEXTWORLD_PROMPT.format(query=query,context=context))

llm = OpenAI(model_name="text-davinci-003",
             openai_api_key=OPENAI_API_KEY,
             temperature=0,
             max_tokens=1000)

chain = load_qa_chain(llm,chain_type="stuff",prompt=CHAT_PROMPT,verbose=False)
docs = [Document(page_content="Bank customers will not be able to trade in shares and mutual funds through their bank "
                              "account. They will need to open and trading account for trading in the market",
                 metadata={}),
        Document(page_content="Bank customers can open trading account by logging into the banks portal",
                 metadata={})
        ]

while True:
    query = input("What is your question:\n")
    response = chain.run(input_documents=docs,query=query)
    print(response)