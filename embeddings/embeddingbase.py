from abc import abstractmethod, ABC


class Embeddings(ABC):
    def __init__(self):
        self.module = "Embeddings"


    @abstractmethod
    def generate_embeddings(self,text):
        pass
