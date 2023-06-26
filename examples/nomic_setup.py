# from nomic.gpt4all import GPT4All
# m=GPT4All()
# m.open()
# resp = m.prompt("Write me a story about a lonely computer")
#
# print(resp)
from langchain import LLMChain, PromptTemplate
from langchain.llms import GPT4All

llm = GPT4All(model='../data/gpt4all-converted.bin')
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

resp=llm_chain.run(question)
print(resp)

