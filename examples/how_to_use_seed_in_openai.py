import os

import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class OpenAIFeatures():
    def __init__(self):
        self.model = "gpt-3.5-turbo-1106"

    def test_seed(self, seed: int = None):

        # Below context taken from https://www.investopedia.com/terms/b/blockchain.asp
        context = "A blockchain is a distributed database or ledger shared among a computer network's nodes. They are best " \
                  "known for their crucial role in cryptocurrency systems for maintaining a secure and decentralized record " \
                  "of transactions, but they are not limited to cryptocurrency uses. Blockchains can be used to make data in " \
                  "any industry immutable—the term used to describe the inability to be altered. "

        system_message = "You are a helpful chat assistant.You answer questions based on provided context"

        user_request_template = "Please answer the question based on provided context only.If answer is not there " \
                       "in context, please politely say that you do not know the answer " \
                       "context: " \
                       "{context}" \
                       "question: " \
                       "{question} "

        question = "How can we use blockchain. Please provide a summarized answer ?"
        user_request = user_request_template.format(context=context,question=question)

        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_request},
            ]
            if seed is None:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7,
                    api_key=OPENAI_API_KEY
                )
            else:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=200,
                    seed=seed,
                    temperature=0.7,
                    api_key=OPENAI_API_KEY)

            response_content = response["choices"][0]["message"]["content"]
            system_fingerprint = response["system_fingerprint"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = (
                    response["usage"]["total_tokens"] - response["usage"]["prompt_tokens"]
            )

            return response_content, system_fingerprint, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def test_vision_model(self):
        pass


if __name__ == "__main__":
    open_ai_features = OpenAIFeatures()

    for i in range(5):
        response_content, system_fingerprint, prompt_tokens, completion_tokens = open_ai_features.test_seed(None)
        print("Response :")
        print(response_content)
        print("system_fingerprint :")
        print(system_fingerprint)
        print("prompt tokens :")
        print(prompt_tokens)
        print("completion tokens :")
        print(completion_tokens)


