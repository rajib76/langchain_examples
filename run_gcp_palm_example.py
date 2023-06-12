import os

from dotenv import load_dotenv
from langchain.llms import GooglePalm


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

class GCPLLM():
    def __init__(self):
        self.module = "PALM"

    def get_response(self,query,model="models/text-bison-001"):
        llm = GooglePalm(model_name=model,
                 temperature=0)
        response = llm(query)
        print(response)

if __name__=="__main__":
    gcp_llm = GCPLLM()
    query = "Who is Indira Gandhi?"
    gcp_llm.get_response(query)

