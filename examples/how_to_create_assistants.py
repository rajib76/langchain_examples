import os

import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def deploy_assistant():
    file = openai.files.create(
        file=open("/Users/joyeed/langchain_examples/langchain_examples/data/faq/covid_faq.pdf", "rb"),
        purpose='assistants'
    )

    # Add the file to the assistant
    assistant = openai.beta.assistants.create(
        name="covid_faq",
        instructions="You are a covid support chatbot. Use your knowledge base to best respond to patient queries.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id]
    )

    return file, assistant


def run_assistant(file_id, assistant_id,question):
    thread = openai.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": question,
                "file_ids": [file_id]
            }
        ]
    )

    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    return run,thread


def get_answer(run, thread):
    print("Looking for an answer to your question...")
    while run.status != "completed":
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    print("Phew, that was a lot of reading...")
    messages = openai.beta.threads.messages.list(
        thread_id=thread.id
    )

    # print(messages.data[0].content[0].text.value)
    annotations = messages.data[0].content[0].text.annotations
    message_content = messages.data[0].content[0].text.value
    return annotations,message_content


if __name__ == "__main__":
    # file,assistant = deploy_assistant()
    # print(file)
    # print(assistant)
    # question = "Is specimen collection covered by medicare?"
    while True:
        question = input("How may I help you today? \n")
        if "exit" in question.lower():
            break

        run,thread = run_assistant("file-joHFujqwo6pBZkDvqsg4Yy2e","asst_9fCkaPHl0KzKX3B3TJ2gddBF",question)
        annotations,message_content = get_answer(run,thread)
        print(message_content)
    print("Thanks and happy to serve you")


# message = openai.beta.threads.messages.retrieve(
#   thread_id="thread_XpKtOiPf84p5AaDS4gzoOPaU",
#   message_id="run_Hdo37FAy57I481mpmPUCFKY8"
# )

# print(message)
