from langchain import PromptTemplate

from llm_prompt_template.prompt_templates import PromptTemplates


class GuidedTemplate(PromptTemplates):
    def __init__(self):
        super().__init__()
        self.module = "GuidedTemplate"

    def get_prompt(self):
        prompt = PromptTemplate(input_variables=["question", "answer","context","chat_history"], template="Answer based on the context only. Please be very specific and to the point. Answer only if it can be fully answered based on the context Context:{context}\n{chat_history}\nQuestion: {question}\n{answer}")
        return prompt