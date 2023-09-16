import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

prompt = hub.pull("rajib76/citation")

context = """
[DOCUMENT] \n\n document_name:History of India \n\n document_text:The history of independent India began when the country became an independent nation within the British Commonwealth on 15 August 1947. Direct administration by the British, which began in 1858, affected a political and economic unification of the subcontinent. When British rule came to an end in 1947, the subcontinent was partitioned along religious lines into two separate countries—India, with a majority of Hindus, and Pakistan, with a majority of Muslims.[1] Concurrently the Muslim-majority northwest and east of British India was separated into the Dominion of Pakistan, by the Partition of India. \n\n 
[END DOCUMENT]
\n\n
[DOCUMENT] \n\n document_name:Vedic period \n\n document_text:The Vedic period, or the Vedic age (c. 1500 – c. 500 BCE), is the period in the late Bronze Age and early Iron Age of the history of India when the Vedic literature, including the Vedas (c. 1500–900 BCE), was composed in the northern Indian subcontinent, between the end of the urban Indus Valley Civilisation and a second urbanisation, which began in the central Indo-Gangetic Plain c. 600 BCE. The Vedas are liturgical texts which formed the basis of the influential Brahmanical ideology, which developed in the Kuru Kingdom, a tribal union of several Indo-Aryan tribes. The Vedas contain details of life during this period that have been interpreted to be historical[1][note 1] and constitute the primary sources for understanding the period. These documents, alongside the corresponding archaeological record, allow for the evolution of the Indo-Aryan and Vedic culture to be traced and inferred.[2]. \n\n [END DOCUMENT]
"""

chain_from_hub = prompt | ChatOpenAI(model="gpt-4")
for chunk in chain_from_hub.stream({"question": "What is Kuru Kingdom?",
                           "context":context}):
    print(chunk.content, end="")