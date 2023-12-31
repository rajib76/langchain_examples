import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

name = 'John'

template = '''
Your name is RhyBot and the user's name is {user_name}.
You are a friendly and humorous chatbot.
Your replies are short and funny.
Use informal words to converse with the user.
Avoid being too polite. Just talk casually.

{chat_history}
Human: {human_input}
'''

prompt = PromptTemplate(
    input_variables=['chat_history', 'user_name', 'human_input'],
    template=template
)

memory = ConversationBufferMemory(memory_key='chat_history',input_key='human_input', return_messages=True)
model = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=256, verbose=True
)
llmchain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory
)

def get_response(msg):
    response = llmchain.run({"human_input":msg,"user_name":name})
    return response

while (True):
    msg = input('You: ')
    if msg == 'q':
        break
    print(f'Bot: {get_response(msg)}')