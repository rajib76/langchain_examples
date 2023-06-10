from langchain import PromptTemplate

from lang_output_formatter.example_output import ExampleOutput
from llm_prompt_template.prompt_templates import PromptTemplates


class ExampleTemplateChain(PromptTemplates):
    def __init__(self):
        super().__init__()
        self.module = "ExampleTemplateChain"

    def get_prompt(self):
        example_output = ExampleOutput()
        example_parser = example_output.get_formatted_output()
        format_instructions = example_parser.get_format_instructions()

        prompt = PromptTemplate(
            template="Summarize the content based on the context provided.\n{format_instructions}\nContext: {context}\n Question:{question}",
            input_variables=["question","context"],
            partial_variables={"format_instructions":format_instructions}

        )

        return prompt, example_parser