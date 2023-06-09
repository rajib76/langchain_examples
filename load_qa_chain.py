import os

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.schema import Document

from llm_prompt_template.load_qa_prompt_template import LoadQATemplate

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(model_name="text-davinci-003",
                 openai_api_key=OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=1000)

docs = [Document(page_content="Hi I am from California. I like to eat icecream",metadata={})]

from langchain.chains.question_answering import load_qa_chain
query = "Where do I live?"
prompt_template = LoadQATemplate()
prompt = prompt_template.get_prompt()
print(prompt)
chain = load_qa_chain(llm, chain_type="stuff",prompt=prompt)
resp = chain.run(input_documents=docs, question=query, tone="Sad")

print(resp)