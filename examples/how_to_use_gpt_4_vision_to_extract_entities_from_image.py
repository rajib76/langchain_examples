# Author : Rajib Deb
# Description: Takes and image and extracts entities from the image
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic.v1 import BaseModel

# Getting the API KEY
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


class ImageQA(BaseModel):
    model: str = "gpt-4-vision-preview"

    def image_encode(self, image_source: str):
        """ This function automatically determines the MIME type and encodes the image to base64."""
        mime_type, _ = mimetypes.guess_type(image_source)
        if mime_type is None:
            raise ValueError(f"Mime type determination failed for {image_source}")

        with open(image_source, "rb") as image_content:
            encoded_image_string = base64.b64encode(image_content.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_image_string}"

    def ask_imagae(self, question, image_url):
        """
        This method is where we are passing the base64 encoded content of the image
        :param question: The prompt from the user
        :param image_url: base64 encoded image content
        :return: The response of the use question
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url,
                                          "detail": "auto", }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response


if __name__ == "__main__":
    img = ImageQA()
    question = """I am providing you an image of a judgement. 
    You need to extract the below entities from the image.
    
    case_identifier
    appellant_name
    respondent_name
    judge_name
    judgement_date
    
    There can be multiple applellants and respondants.
    Please output the response in JSON format.
    
    here is an example of the output:
    {
      "case_identifier": "CIVIL APPEAL NO. 2768 OF 2023",
    "appellant_name": ["Rajib Deb","Joe Adam"]
    "respondent_name": ["Mohan kumar" "Sunil Sen"]
    "judge_name": "Ajay Gupta",
    "judgement_date": "05/21/2023"
    }
    """

    image_source = "/Users/joyeed/langchain_examples/langchain_examples/data/ner/Judgement_14-Sep-2023.jpg"
    image_url = img.image_encode(image_source)
    response = img.ask_imagae(question, image_url)
    print(response.choices[0].message.content)
