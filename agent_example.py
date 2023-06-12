import os
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_SESSION"] = "agent_chain"

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory

from lang_example_tools.example_tool import ExampleTool
from llm_prompt_template.example_template import ExampleTemplate

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def main():
    llm = OpenAI(model_name="text-davinci-003",
                 openai_api_key=OPENAI_API_KEY,
                 temperature=0,
                 max_tokens=1000)

    et = ExampleTemplate()
    prompt,out_parser = et.get_prompt()
    _input = prompt.format_prompt(question="What is the salary of Ranjit in the database?")

    memory = ConversationBufferMemory(memory_key="chat_history",input_key="input",output_key="output")

    tools =[ExampleTool()]

    example_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=1,
        early_stopping_method="generate",
        max_execution_time=1,
        memory=memory
        )
    with get_openai_callback() as cb:
        response = example_agent.run({
            "input":_input.to_string()

        })

        print(response)

if __name__=="__main__":
    main()