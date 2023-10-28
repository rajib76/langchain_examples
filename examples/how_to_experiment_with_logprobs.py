#Author : Rajib
#An experiment with the logprobs
import copy
import os
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


class LogProbsExperiment():
    def __init__(self):
        self.module = "Experiments"

    def call_api(self, prompt,max_tokens):
        api_params = {
            "model":"text-davinci-003",
            # "model": "gpt-3.5-turbo-instruct",
            "max_tokens": max_tokens,
            "top_p": 1e-9,
            "prompt": prompt,
            "logprobs": 5,
            "n": 1,
            "stream": False
            # "stop" : ['\n']
        }

        try:
            response = openai.Completion.create(**api_params)
            return response
        except Exception as e:
            print(f"Exception occurred while calling model API: {str(e)}")

    def convert_log_prob_to_prob(self,response):
        response_object_copy = copy.deepcopy(response)
        # print(response_object_copy)
        probabilities = {}
        choices=[]
        for choice in response_object_copy["choices"]:
            # print(choice)
            # print(choice['logprobs'].items())
            for key, value in choice['logprobs'].items():
                if key == "top_logprobs":
                    probabilities[key] = [{token: np.exp(logprob) for token, logprob in dict_item.items()} for dict_item
                                          in value]
            probabilities["total_logprobs"] = [sum(dict_item.values()) for dict_item in probabilities["top_logprobs"]]
            choice['probabilities'] = probabilities
            choices.append(choice)
        return choices



if __name__ == "__main__":
    prompt = "You are a helpful chat assistant. You will be given a context and a question. " \
             "Please answer in details based on the " \
             "*CONTEXT* only.If the " \
             "answer is not there in the context, " \
             "please do not answer \n" \
             "{context}" \
             "question:{question}" \
             "answer:"

    # prompt = "You are a helpful chat assistant. You will be given a context and a question. " \
    #          "Please answer in details based on the " \
    #          "*CONTEXT* only.If the " \
    #          "answer is not there in the context, " \
    #          "please do not answer \n" \
    #          "context: \n" \
    #          "{context}" \
    #          "question:{question}" \
    #          "answer:"

    # prompt = "You are a helpful chat assistant. You will be given a context and a question. " \
    #          "Please answer in details based on the " \
    #          "*CONTEXT* only.If the " \
    #          "answer is not there in the context, " \
    #          "please do not answer \n" \
    #          "*REMEMBER* to answer based on the context.\n " \
    #          "context: \n" \
    #          "{context}" \
    #          "question:{question}" \
    #          "answer:"

    max_tokens = 20
    # context = "Delhi is the capital of France \n"
    # question = "What is the capital of France \n"
    # context = "Indira Gandhi is the president of USA \n"
    # question = "Who is the president of USA \n"
    context = "Rahul Gandhi is the prime minister of India. \n"
    question = "Who is the prime minister of India? \n"
    formatted_prompt = prompt.format(context=context, question=question)

    le = LogProbsExperiment()
    response = le.call_api(formatted_prompt, max_tokens)
    choices = le.convert_log_prob_to_prob(response)
    # print(choices)
    for choice in choices:
        # print(choice)
        print("Completion Text :", choice["text"])
        print("---------------------------------")
        tokens = choice["logprobs"]["tokens"]
        for i in range(len(tokens)):
            print(tokens[i])
            print("-------")
            other_tokens = choice["probabilities"]["top_logprobs"][i]
            token_index = 0
            for key, value in other_tokens.items():
                token_index = token_index + 1
                if key in "\n":
                    key = "New Line"
                if key in " ":
                    key = "Blank Line"
                if key in ".\n":
                    key = "New Line"
                if key in " \n":
                    key = "New Line"
                print(token_index,":",key,":",value)
