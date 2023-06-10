from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import BaseOutputParser

from lang_output_formatter.output_formatter import OutputFormatter


class ExampleOutputChain(OutputFormatter):
    def __init__(self):
        super().__init__()
        self.module = "Example Output Chain"

    def get_formatted_output(self) -> BaseOutputParser:
        """
        This retruns the output parser
        :return: output parser
        """
        response_schemas = [
            ResponseSchema(name="answer",description= "answer to user question"),
            ResponseSchema(name="source", description="source of the answer")

        ]

        out_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        return out_parser