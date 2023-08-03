import os

import uvicorn
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import PostgresChatMessageHistory
from langchain.schema import Document
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

app = FastAPI(title="My generator",
              summary="generator",
              description="Generates response")

# Adds signed cookie-based HTTP sessions. Session information is readable but not modifiable.
# Access or modify the session data using the request.session dictionary interface.
app.add_middleware(SessionMiddleware, secret_key="my-key")


def get_message_history_db(session_id,db="postgres"):
    # This can be made as an utility function to be reused
    # by any other modules
    if db.lower() == "postgres":
        connection_string = "postgresql://admin:password@localhost/postgres"
        message_history = PostgresChatMessageHistory(session_id=session_id, connection_string=connection_string)

    return message_history


@app.get("/generate/", tags=["Answer Generation"])
async def generate_session_id(request: Request,query:str) -> RedirectResponse:
    # In this route,I am creating the session id cookie and the refirecting to
    # the actual endpoint
    params = request.query_params
    session_id = request.session.get("session_id", None)
    if session_id is None:
        request.session["session_id"]="1234"
    return RedirectResponse("/generate_resp/?{params}".format(params=params))


@app.get("/generate_resp/", tags=["Answer Generation"])
async def get_response(request: Request,query:str):
    # The is the actual response generation part

    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-0613',
        openai_api_key=OPENAI_API_KEY
    )

    session_id = request.session.get("session_id", None)
    print(session_id)
    message_history = get_message_history_db(session_id,"postgres")
    prompt = PromptTemplate(
        template="Answer the user question based on provided context only and history. Do not answer anything which is not in the context"
                 "\n\nHistory:{history}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["question", "context", "history"]
    )

    conversation = load_qa_chain(
        llm=llm,
        prompt=prompt,
        chain_type="stuff",
        verbose=True
        # kwargs={"memory": memory}
    )

    docs = [Document(page_content="Python is a fun language to learn. It is one of the most popular computer language")]
    response = conversation.run(input_documents=docs, question=query, history=str(message_history.messages))
    print(response)
    message_history.add_user_message(query)
    message_history.add_ai_message(response)
    return response


if __name__=="__main__":
    uvicorn.run(host="127.0.0.1",port=8000,app=app)



