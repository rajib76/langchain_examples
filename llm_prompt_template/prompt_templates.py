from abc import ABC, abstractmethod


class PromptTemplates(ABC):
    def __init__(self):
        self.module = "PromptTemplated"

    @abstractmethod
    def get_prompt(self):
        pass
