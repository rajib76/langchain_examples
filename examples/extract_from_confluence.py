import os

from dotenv import load_dotenv
from langchain.document_loaders import ConfluenceLoader

load_dotenv()
# You need to create the api token from https://id.atlassian.com/manage-profile/security/api-tokens
# the space key can be found under space settings->manage space->key
# the url is your <your-account>.atlassian.net/wiki
# user name is your email
confluence_api_key = os.environ.get('confluence_api_key')
confluence_user_name = os.environ.get('confluence_user_name')
confluence_page_key = os.environ.get('confluence_page_key')
confluence_url = os.environ.get('confluence_url')

loader = ConfluenceLoader(url=confluence_url,username=confluence_user_name,api_key=confluence_api_key)

documents = loader.load(space_key=confluence_page_key, include_attachments=True, limit=50)
print(documents)