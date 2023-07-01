# Extract life events from texts
# https://www.healthcare.gov/glossary/qualifying-life-event/
import os

from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Load Environment Variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Create Examples
examples = [
    {
        "context":"Sumana had a baby yesterday",
        "query": "What is the life event?",
        "answer": "Life event is 'new born'"
    },
    {
        "context": "Archana is expecting a baby in next 7 days",
        "query": "What is the life event?",
        "answer": "Life event is 'new born'"
    },
    {
        "context": "Mohan lost his high paying job",
        "query": "What is the life event?",
        "answer": "Life event is 'losing a job'"
    },
    {
        "context": "Kavita celebrated her 26th birthdaya today",
        "query": "What is the life event?",
        "answer": "Life event is 'turning 26'"
    },
    {
        "context": "Ram is relocating to Sri Lanka",
        "query": "What is the life event?",
        "answer": "Life event is 'moving to a new location'"
    }

]

fewshot_prompt_template = """
Context:{context}
User: {query}
AI: {answer}
"""

fewshot_prompt = PromptTemplate(
    input_variables=["context","query","answer"],
    template = fewshot_prompt_template
)

# Create the prefix

prefix = """The following are exerpts from comversation with an AI assistant
who understands life events. Please ensure that you are correctly classifying a life event.
Life events are a change of a situation in someone's life and only the below scenarios are applicable
to consider the event as a life event

    - Losing existing health coverage, including job-based, individual, and student plans
    - Losing eligibility for Medicare, Medicaid, or CHIP
    - Turning 26 and losing coverage through a parent’s plan
    - Getting married or divorced
    - Having a baby or adopting a child
    - Death in the family
    - Moving to a different ZIP code or county
    - A student moving to or from the place they attend school
    - A seasonal worker moving to or from the place they both live and work
    - Moving to or from a shelter or other transitional housing
    - Changes in your income that affect the coverage you qualify for
    - Gaining membership in a federally recognized tribe or status as an Alaska Native Claims Settlement Act (ANCSA) Corporation shareholder
    - Becoming a U.S. citizen
    - Leaving incarceration (jail or prison)
    - AmeriCorps members starting or ending their service

Here are the examples
"""

suffix ="""
Context:{context}
User:{query}
AI: 
"""

# Create the few shot prompt

few_shot_prompt_template = FewShotPromptTemplate(
    examples = examples,
    example_prompt = fewshot_prompt,
    prefix = prefix,
    suffix=suffix,
    input_variables =["context","query"],
    example_separator = "\n\n"
)

query = "What is the life event in the provided context"
# context ="Manorama gave birth to a baby at 8 am today"
#
# print(few_shot_prompt_template.format(query=query,context=context))

llm = OpenAI(model_name="text-davinci-003",
             openai_api_key=OPENAI_API_KEY,
             temperature=0,
             max_tokens=1000)

chain = load_qa_chain(llm,chain_type="stuff",prompt=few_shot_prompt_template)
docs = [Document(page_content="Manorama went running this morning",metadata={})]
response = chain.run(input_documents=docs,query=query)

print(response)

















