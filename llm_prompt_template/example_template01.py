from langchain import PromptTemplate

from llm_prompt_template.prompt_templates import PromptTemplates


class ExampleTemplate01(PromptTemplates):
    def __init__(self):
        super().__init__()
        self.module = "LoadQATemplate"

    def get_prompt(self):
        prompt = PromptTemplate(
            template="Answer the user question based on provided context. Ensure to answer in the desired format. "
                     "List each event separately"
                     "Desired format: "
                     "date:  date of the event"
                     "event: name of the event"
                     "\n\nContext: {context}\n\n Question: {question}\n\nAnswer:",
            input_variables=["question","context"]
        )

        return prompt