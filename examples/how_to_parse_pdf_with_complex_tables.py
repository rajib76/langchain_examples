# Author : Rajib
# This program shows how to do context aware parsing of a large PDF and then summarize it
# References:
# https://python.langchain.com/docs/use_cases/summarization
# https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/
# https://smith.langchain.com/hub/
import json
import logging
import os
import re
import zipfile

import pandas as pd
import pinecone
from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import LLMChain, StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



class PDFExtract():
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        # initialize pinecone
        # One thing I dound out(may be a defect in PINECONE), the api_key and the environment must be provided as below
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
            environment=os.getenv("PINECONE_ENV"),  # next to api key in console
        )

    def _get_credentials(self):
        credentials = Credentials.service_principal_credentials_builder().with_client_id(
            self.client_id).with_client_secret(self.client_secret).build()

        return credentials

    def _zip_file(self, output_path):
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

    def _parse_json(self, json_file_path):
        with open(json_file_path, "r") as json_file:
            content = json.loads(json_file.read())

        pdf_element = content["elements"]
        return pdf_element

    def load_pine_index(self,docs,index_name = "arxiv-index"):

        emb = OpenAIEmbeddings()
        docsearch = Pinecone.from_documents(docs, emb, index_name=index_name)

    def get_files_from_dir(self, dir):
        file_list = []
        files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

        return files

    def load_docs(self, file_path):
        loader = TextLoader(file_path)
        docs = loader.load()

        return docs

    def parse_pdf(self, input_file_path, output_path, unzip_dir, chunked_dir):
        try:
            credentials = self._get_credentials()
            execution_context = ExecutionContext.create(credentials)
            extract_pdf_operation = ExtractPDFOperation.create_new()
            source = FileRef.create_from_local_file(input_file_path)
            extract_pdf_operation.set_input(source)
            extract_pdf_operation.set_input(source)
            extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
                .with_element_to_extract(ExtractElementType.TEXT) \
                .with_element_to_extract(ExtractElementType.TABLES) \
                .build()
            extract_pdf_operation.set_options(extract_pdf_options)

            # Execute the operation.
            result: FileRef = extract_pdf_operation.execute(execution_context)

            # Save the result to the specified location.
            result.save_as(output_path)
            self._zip_file(output_path)
            json_file_path = os.path.join(unzip_dir, "structuredData.json")
            elements = self._parse_json(json_file_path)

            file_split = 0
            # Define the header flag. If first time header no need to cut a new file
            FIRST_TIME_HEADER = True
            file_name = os.path.join(chunked_dir, f"file_{file_split}".format(file_split=file_split))
            parsed_file = open(file_name, "a", encoding="utf-8")
            for element in elements:
                if "//Document/H2" in element["Path"]:
                    hdr_txt = element["Text"]
                    if FIRST_TIME_HEADER:
                        FIRST_TIME_HEADER = False
                        parsed_file.write(hdr_txt)
                        parsed_file.write("\n")
                    else:
                        parsed_file.close()
                        file_split = file_split + 1
                        file_name = os.path.join(chunked_dir, f"file_{file_split}".format(file_split=file_split))
                        parsed_file = open(file_name, "a", encoding="utf-8")
                        parsed_file.write(hdr_txt)
                        parsed_file.write("\n")
                else:
                    if "Document/Table" in element["Path"]:
                        match = re.search(r'^//Document/Table(?:\[\d+\])?$', element["Path"])
                        if match:
                            xlsx_file_name = element["filePaths"][0]
                            xlsx_file = os.path.join(unzip_dir, xlsx_file_name)
                            df = pd.DataFrame(pd.read_excel(xlsx_file))
                            table_content = df.to_markdown().replace("_x000D_", "      ")
                            parsed_file.write(table_content)
                            parsed_file.write("\n")
                    else:
                        try:
                            text_content = element["Text"]
                            parsed_file.write(text_content)
                            parsed_file.write("\n")
                        except KeyError as ke:
                            pass
            parsed_file.close()
        except Exception as e:
            print(e)
            logging.exception("Exception encountered while executing operation")


if __name__ == "__main__":
    load_dotenv()
    pdf_extract_client_id = os.getenv("pdf_extract_client_id")
    pdf_extract_client_secret = os.getenv("pdf_extract_client_secret")

    input_file_path = "/Users/joyeed/langchain_examples/langchain_examples/data/long_context.pdf"
    output_path = "/Users/joyeed/langchain_examples/langchain_examples/data/pdf/long_context.zip"
    unzip_dir = "/Users/joyeed/langchain_examples/langchain_examples/data/pdf"
    chunked_dir = "/Users/joyeed/langchain_examples/langchain_examples/data/pdf/chunks"
    isExist = os.path.exists(chunked_dir)
    if not isExist:
        os.makedirs(chunked_dir)
    pdf_extract = PDFExtract(pdf_extract_client_id, pdf_extract_client_secret)

    # Step - 1 : Run this step to chunk the PDF into contextual subsections

    # pdf_extract.parse_pdf(input_file_path,output_path,unzip_dir,chunked_dir)

    # Step - 2 : use a TextLoader to get all the chunks in a list of Dcoments

    files = pdf_extract.get_files_from_dir(chunked_dir)
    print(files)
    list_of_all_docs=[]
    for file in files:
        document = pdf_extract.load_docs(file)
        print(document)
        list_of_all_docs.append(document[0])


    # pdf_extract.load_pine_index(list_of_all_docs)

    # Step - 3 : Summarization process starts from here

    # Instantiating the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Getting the map prompt from hub
    map_prompt = hub.pull("rajib76/map_chain")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Getting the reduce prompt from hub
    reduce_prompt = hub.pull("rajib76/reduce_chain")
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=16000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="documents",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    print(map_reduce_chain.run(list_of_all_docs))


