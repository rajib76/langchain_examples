# The langchain AWS S3 loader does not provide
# a way to specify the access keys within the program like as below
# s3_client = boto3.resource('s3',
#                       aws_access_key_id='<your_access_key_id>',
#                       aws_secret_access_key='<your_secret_access_key>',
#                       region_name='us-east-1'
#                       )
# So, you will need to create a config and credentials under .aws folder in root directory. Follow the below on MAC
# 1. touch ~/.aws/config - the above command will create a config file.
# 2. cd ~/.aws
# 3. vi config
# 4. add the below in the config - region should be the region where you created the file
# [default]
# region = us-east-1
# After this create a credentials file
# 1. touch ~/.aws/credentials
# 2 cd ~/.aws
# 3. vi credentials
# 4. add the below in the credentials
# [default]
# aws_access_key_id = <your access key>
# aws_secret_access_key = <your secret key>
# If windows, create these files under your home directory c:/Users/<user_name>/.aws
# other option is creating custom loader - see document_loading_from_S301.py

from langchain.document_loaders import S3DirectoryLoader

loader = S3DirectoryLoader("langchain-bucket01")
data = loader.load()

print(data)
