import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import openai
import requests
from PIL import Image
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY,max_tokens=1500)

class MyImageGenTool(BaseTool):
    name = "GenerateImage"
    description = "Useful for when there is a need to generate an image." \
                  "Input: A prompt describing the image " \
                  "Output: Only the url of the generated image"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Image.create(
            prompt=query,
            n=1,
            size="256x256",
        )

        return response["data"][0]["url"]

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("Does not support async")


def url_find(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


if __name__=="__main__":
    tools = [MyImageGenTool()]

    mrkl = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    output = mrkl.run("I want to market a new credit card for my bank.The credit card is suitable for students.Generate a photo-realistic image which can be used to market this credit card. Please output the url of the image.")

    print(output)

    image_url = url_find(output)
    print(image_url)
    response = requests.get(image_url[0], stream=True)
    img = Image.open(response.raw)

    plt.imshow(img)
    plt.show()