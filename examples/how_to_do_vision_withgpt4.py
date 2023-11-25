# Author : Rajib Deb
# Description: Takes and image and does Q&A over the image
# We will see how we can use this to create product descriptions
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic.v1 import BaseModel

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
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url,
                                          "detail": "high", }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        # response = client.chat.completions.create(
        #     model="gpt-4-vision-preview",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": question},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": image_url,
        #                 },
        #             ],
        #         }
        #     ],
        #     max_tokens=300,
        # )

        return response


if __name__ == "__main__":
    img = ImageQA()
    # question = "Please create a caption of the image that can be used to search for the item in a shopping portal"
    # question = "Please create a detailed caption of the kitchen island that can be used for searching in a e-commerce " \
    #            "portal in json format"

    # question = "Please create a detailed caption of the image that can be used for searching in an e-commerce " \
    #            "portal in json format"

    question = "I am providing you an image of my refrigerator. Please tell me what do I have in my refrigerator and what " \
               "needs to be replenished. "

    # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    # image_source = "/Users/joyeed/langchain_examples/langchain_examples/data/images/island.jpg"
    # image_source = "/Users/joyeed/langchain_examples/langchain_examples/data/images/chandelier_.jpg"
    image_source = "/Users/joyeed/langchain_examples/langchain_examples/data/images/my_freeze_1.jpeg"
    image_url = img.image_encode(image_source)
    response = img.ask_imagae(question, image_url)
    print(response.choices[0].message.content)
