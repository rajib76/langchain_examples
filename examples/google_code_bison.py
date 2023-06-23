# Source : https://python.langchain.com/docs/modules/model_io/models/llms/integrations/google_vertex_ai_palm
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.llms import VertexAI

# text-bison
# textembedding-gecko
# chat-bison
# code-bison
# codechat-bison
# code-gecko
# https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models


load_dotenv()

llm = VertexAI(model_name="code-bison")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Write a python function that converts ascii to ebcdic?"

resp = llm_chain.run(question)

print(resp)