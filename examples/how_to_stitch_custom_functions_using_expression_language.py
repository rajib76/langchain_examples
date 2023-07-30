import ast
import json
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# By default it uses gpt-3.5-turbo
entity_extraction_model = ChatOpenAI()
# I can use any other model also
email_writing_model = ChatOpenAI(model="gpt-3.5-turbo-16k")
#email_writing_model = OpenAI(model="text-davinci-003")


def get_entities(text):
    """
    This module extracts the required entities from the text passed to it
    using langchain's create_extraction_chain
    :param text: The text from which the entities need to be extracted
    :return: the extracted entities
    """

    # Here I am defining the schema
    schema = {
        "properties": {
            "person": {"type": "string"},
            "preference": {"type": "string"}
        },
        "required": ["person", "preference"],
    }

    chain = create_extraction_chain(schema, entity_extraction_model)

    resp = chain.run({"input": text})

    # The create_extraction chain returns multiple sets of extracted entities.
    # But for the purpose of the demo, I am returing only one occurence of the
    # extracted entity
    return resp[0]


def get_entities_prod_description(text):
    """
    This module extracts the product description which matches the
    preference of the user and then adds it to the returned entities
    {"person":"Rajib","preference":"coffee","description":"We have the best coffee..."
    :param text: The text from which entities need to be extracted
    :return: The extracted entities
    """
    entities = get_entities(text)
    # I have harcoded the product description here, because I was lazy.
    # But it can be made more interesting by doing a similarity
    # search between the preference and product description
    prod_desc = "We have the best coffee from the land of Africa. it is best in Aroma and directly sourced from farmers so that we can offer that to you at the least price but with the highest quality"
    entities["description"] = prod_desc

    return ast.literal_eval(json.dumps(entities))


if __name__=="__main__":
    # This is my prompt to create the email for my coffee sale campaign
    prompt = ChatPromptTemplate.from_template("Write a marketing email to {person} based on "
                                              "person's {preference} and product description {description} ")
    # I extract the named entities from the text I pass
    # This text is hardcoded, but it can be extracted by mining social media
    # which will have the preferences and behavior of our customers
    ner_chain = {"text": itemgetter("text")} | RunnableLambda(get_entities_prod_description)
    # The email chain is now composed with the ner_chain, the prompt and the model I want to use
    # to create the email
    email_chain = ner_chain | prompt | email_writing_model
    resp = email_chain.invoke({"text": "Rajib Loves Coffee"})
    #See how the email now looks
    print(resp.content)
