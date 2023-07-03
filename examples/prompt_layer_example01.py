import os
from typing import Dict, Any, Tuple, List

import promptlayer
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import PromptLayerOpenAI
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string

load_dotenv()
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
prompt_layer_api_key = os.environ.get('prompt_layer_api_key')

openai = promptlayer.openai
openai.api_key = os.environ.get("OPENAI_API_KEY")
promptlayer.api_key = prompt_layer_api_key


# My custom conversation memory

class MyConversationBufferMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def _get_input_output(
            self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            print("Inputs are {inputs}".format(inputs=inputs))
            # prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
            prompt_input_keys = list(set(inputs).difference(self.memory_variables + ["stop"]))
            prompt_content = ""
            for key in prompt_input_keys:
                prompt_content = prompt_content + "\n\n" + str(inputs[key])
            prompt_input_key = "input_content"
            inputs = {prompt_input_key: prompt_content}
            print("Final Inputs are {inputs}".format(inputs=inputs))
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)


class ChatWithPromptLayer():
    def __init__(self):
        self.module = "Chat with Prompt Layer"
        llm = PromptLayerChatOpenAI(pl_tags=["load_qa_chain"], return_pl_id=True, model_name='gpt-3.5-turbo-0613')
        memory = MyConversationBufferMemory()
        self.llm = llm
        self.memory = memory

    def get_response(self):
        history = self.memory.chat_memory.messages
        prompt = PromptTemplate(
            template="Answer the user question based on provided context and history only."
                     "\n\nHistory:{history}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question", "context", "history"]
        )

        conversation = load_qa_chain(
            llm=self.llm,
            prompt=prompt,
            chain_type="stuff",
            verbose=True,
            # kwargs={"memory": memory}
            memory=self.memory
        )

        return conversation


if __name__ == "__main__":
    cpl = ChatWithPromptLayer()
    convo = cpl.get_response()
    loader = PyMuPDFLoader("../data/pdf-test.pdf")
    data = loader.load()
    while True:
        inpt = input("You: ")
        resp = convo.run(input_documents=data, question=inpt, history=str(cpl.memory.chat_memory))
        print(resp)
