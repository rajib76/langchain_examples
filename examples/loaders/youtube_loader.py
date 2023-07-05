# https://ffbinaries.com/downloads
# download ffmpeg and ffprobe from the above link - For MAC use osx-64
# I then copied it in the bin folder(it can be any folder in the PATH)
# I also need to provide permission by running
# chmod -R 777 ffprobe
# chmod -R 777 ffmpeg
import os

from dotenv import load_dotenv
from langchain.document_loaders import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class YouTubeLoader(BaseModel):
    url: str
    tgt_dir: str

    def load_docs(self):
        y_loader = GenericLoader(
            YoutubeAudioLoader([self.url], self.tgt_dir),
            OpenAIWhisperParser()
        )

        docs = y_loader.load()

        return docs

    def __call__(self, *args, **kwargs):
        docs = self.load_docs()

        return docs


if __name__ == "__main__":
    url = "https://youtu.be/IkXwfhk544g"
    tgt_dir = "../youtube/"
    ytb_loader = YouTubeLoader(url=url, tgt_dir=tgt_dir)
    docs = ytb_loader()
    print(docs)
