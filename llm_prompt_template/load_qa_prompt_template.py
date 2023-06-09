from langchain import PromptTemplate

from lang_output_formatter.example_output import ExampleOutput
from llm_prompt_template.prompt_templates import PromptTemplates


class LoadQATemplate(PromptTemplates):
    def __init__(self):
        super().__init__()
        self.module = "LoadQATemplate"

    def get_prompt(self):
        prompt = PromptTemplate(
            template="Answer the user question based on provided context. Ensure to answer in the provided tone. "
                     "For happy tone use a smiley. For other tones use an appropriate emoji"
                     "\n\nContext: {context}\n\n Tone: {tone}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question","context","tone"]
        )

        return prompt