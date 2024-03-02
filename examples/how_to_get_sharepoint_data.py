import os

from dotenv import load_dotenv
from langchain_community.document_loaders.sharepoint import SharePointLoader
load_dotenv()
O365_CLIENT_ID = os.environ.get('O365_CLIENT_ID')
O365_CLIENT_SECRET = os.environ.get('O365_CLIENT_SECRET')

loader = SharePointLoader(document_library_id="b!iwb46tJIgE-1t1W36fyDEpJifqUam5RDlRATDJPFjxi-whHLeOPQT4l_O6LYX7SB", folder_path="/documents", auth_with_token=True)
#
documents = loader.load()
print(documents)