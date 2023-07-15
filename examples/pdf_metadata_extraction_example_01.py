#python -m spacy download en_core_web_sm
import json
import os

import PyPDF2
from dotenv import load_dotenv
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def get_schema():
    schema = {
        "properties": {
            "case_identifier": {"type": "string"},
            "appellant_name(s)": {"type": "array", "items": {"type": "string"}},
            "respondent_name(s)": {"type": "array", "items": {"type": "string"}},
            "judge_name(s)": {"type": "array", "items": {"type": "string"}},
            "judgement_date": {"type": "string"},
        },
        "required": [],
    }

    return schema


def get_all_pages(pdf_loc="./data/ner/judge.pdf"):
    reader = PyPDF2.PdfReader(pdf_loc)
    texts = []
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        texts.append({"page": i, "page_content": text})

    return texts


if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    schema = get_schema()
    chain = create_extraction_chain(schema, llm)
    texts = get_all_pages()
    extracts = []
    for text in texts:
        page = text["page"]
        content = text["page_content"]
        resp = chain.run(content)
        extracts.append({"page": page, "entities": resp})

    print(extracts)
    for extract in extracts:
        print("Page number:", extract.get('page'))
        print("Entities:", json.dumps(extract.get("entities")[0]))

