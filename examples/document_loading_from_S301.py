# Using custom S3 loader
import os
import tempfile
from typing import List

from dotenv import load_dotenv
from langchain.document_loaders import S3DirectoryLoader, S3FileLoader, UnstructuredFileLoader
from langchain.schema import Document

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")


class MyS3FileLoader(S3FileLoader):
    def __init__(self, bucket: str, key: str):
        super().__init__(bucket, key)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import `boto3` python package. "
                "Please install it with `pip install boto3`."
            )
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                          region_name="us-east-1")
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.key}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3.download_file(self.bucket, self.key, file_path)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()


class MyS3DirectoryLoader(S3DirectoryLoader):
    def __init__(self, bucket: str):
        super().__init__(bucket)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                            region_name="us-east-1")
        bucket = s3.Bucket(self.bucket)
        docs = []
        for obj in bucket.objects.filter(Prefix=self.prefix):
            loader = MyS3FileLoader(self.bucket, obj.key)
            docs.extend(loader.load())
        return docs


loader = MyS3DirectoryLoader("langchain-bucket01")
data = loader.load()

print(data)
