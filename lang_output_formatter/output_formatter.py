from abc import ABC, abstractmethod

from langchain.schema import BaseOutputParser


class OutputFormatter(ABC):
    def __init__(self):
        self.module = "OutputFormatter"

    @abstractmethod
    def get_formatted_output(self) -> BaseOutputParser:
        pass

