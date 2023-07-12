import json
import os

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

def csv_analytics(csv):
    '''
    creates a dataframe from csv

    Inputs:
        - csv (str): location of csv
    '''
    df = pd.read_csv(csv)
    print(df.head())

def csv_bar_plot(csv,column):
    import matplotlib.pyplot as plt
    '''
    creates a dataframe from csv

    Inputs:
        - csv (str): location of csv
    '''
    df = pd.read_csv(csv)
    df.plot.bar(x = column)
    plt.show()


my_custom_functions = [
    {
        'name': 'csv_analytics',
        'description': 'Preview data from the dataframe created from csv',
        'parameters': {
            'type': 'object',
            'properties': {
                'csv': {
                    'type': 'string',
                    'description': 'location of the csv'
                }
            }
        }
    },
{
        'name': 'csv_bar_plot',
        'description': 'Create bar plot from the dataframe created from csv',
        'parameters': {
            'type': 'object',
            'properties': {
                'csv': {
                    'type': 'string',
                    'description': 'location of the csv'
                },
                'column': {
                    'type': 'string',
                    'description': 'name of the column in the csv'
                }
            }
        }
    }
]

prompt = "create a bar chart for column Embarked in csv /Users/joyeed/langchain_examples/langchain_examples/data/titanic.csv"
openai_response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = [{'role': 'user', 'content':prompt}],
    functions = my_custom_functions,
    function_call = 'auto'
)['choices'][0]['message']

print(openai_response)

if openai_response.get('function_call'):
    function_called = openai_response['function_call']['name']
    function_args = json.loads(openai_response['function_call']['arguments'])
    eval(function_called)(*list(function_args.values()))

