import os

from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import SystemMessage

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

model = OpenAI(model_name="text-davinci-003",
               openai_api_key=OPENAI_API_KEY,
               temperature=0,
               max_tokens=1000)

# SYSTEM_PROMPT = "You are a helpful chatbot who helps in booking movie tickets online." \
#                 "Please ask only one follow-up question to get the details to book the ticket."


# Meta Language Creation:
# when you would like to generate the output by altering certain notation of the instruction

# SYSTEM_PROMPT = "You are a helpful chatbot who helps in booking movie tickes online. " \
#                 "When I say 'I', I am referring to the {client}. You address by client name" \
#                 "Please ask only one follow-up question to get the details to book the ticket."

# Output Automator:
# when you would like the LLM to generate output in a specified format

# SYSTEM_PROMPT = "You are a helpful chatbot who helps in booking movie tickets online. " \
#                 "When I say 'I', I am referring to the {client}. You address by client name" \
#                 "If the answer consists of multiple steps, generate the steps in seperate bullets and mention the " \
#                 "step name as a bold heading." \
#                 "Please ask only one follow-up question to get the details to book the ticket."

# SYSTEM_PROMPT = "You are a helpful travel agent who guides tourists to travel from USA to India . " \
#                 "If there are alternate ways to travel, please list all alternatives seperately"


# Alternative Approach
# Offer alternative ways of accomplishing a task so that users are aware of all approaches
# SYSTEM_PROMPT = "You are a chef in a restaurant. You answer questions on how to prepare a particular dish."
SYSTEM_PROMPT = "You are a chef in a restaurant. You answer questions on how to prepare a particular dish." \
                "If there are alternate ways to cook, please list both alternatives separately."

HUMAN_PROMPT = "{question}"

system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

llm_chain = LLMChain(llm=model, prompt=chat_prompt)
handler = StdOutCallbackHandler()
# resp = llm_chain.run({"question": "I want to book a ticket"},
#                      callbacks=[handler])
# resp = llm_chain.run({"question": "I want to book a ticket", "client": "rajib"},
#                      callbacks=[handler])
resp = llm_chain.run({"question": "How do I cook fish curry?"},
                     callbacks=[handler])
print(resp)
