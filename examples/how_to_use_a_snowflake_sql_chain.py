import os
from urllib import parse

from dotenv import load_dotenv
from langchain import PromptTemplate, SQLDatabase, OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine, MetaData

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class SnowflakeDBAssist():
    def __init__(self):
        self.module = "SnowflakeDBAssist"

    def get_response(self):
        # sf_prompt_template = "Given an input question and a *dbt* schema and sources, first create a syntactically
        # correct {dialect} query to run, " \ "then look at the results of the query and return the answer. Use the
        # following format. " \ "Question: Question here " \ "SQLQuery: SQL Query to run " \ "SQLResult: Result of
        # the SQLQuery " \ "Answer: Final answer here " \ "Question: {input}"

        # This is the prompt which takes in the source.yml and schema.yml of DBT
        sf_prompt_template = "Given an input question and a *dbt* schema and sources, " \
                             "first create a * syntactically * correct {{dialect}} query to run, " \
                             "then look at the results of the query and return the answer. " \
                             "Provide the answer in the following format. " \
                             "Desired format:" \
                             "Question:\n\n Question here " \
                             "SQLQuery:\n\n SQL Query to run " \
                             "SQLResult:\n\n Result of the SQLQuery " \
                             "Answer:\n\n Final answer here " \
                             " dbtschema: {schema} \n\n \n\n dbtsource:{source}\n\n Question: {{input}}"



        sf_user = os.environ.get('sf_user')
        sf_pass = parse.quote(os.environ.get('sf_pass'))
        sf_schema = os.environ.get('sf_schema')
        sf_db = os.environ.get('sf_db')
        sf_warehouse = os.environ.get('sf_warehouse')

        # engine = create_engine(
        #     'snowflake://{sf_user}:{sf_pass}@jzb87987.prod3.us-west-2/{sf_db}/{sf_schema}?warehouse={sf_warehouse}'.format(
        #         sf_user=sf_user,
        #         sf_pass=sf_pass,
        #         sf_warehouse=sf_warehouse,
        #         sf_schema=sf_schema,
        #         sf_db=sf_db
        #     )
        # )

        # Had to give the sf_schema in the url, else it is failing
        # I thought I should be able to pass it through the MetaData parameter
        # but it is not accepting that in the code. Need to deep dive little more
        engine = create_engine(
            'snowflake://{sf_user}:{sf_pass}@jzb87987.prod3.us-west-2/{sf_db}/{sf_schema}?warehouse={sf_warehouse}'.format(
                sf_user=sf_user,
                sf_pass=sf_pass,
                sf_warehouse=sf_warehouse,
                sf_schema=sf_schema,
                sf_db=sf_db
            )
        )

        # Tried this to pass the schema dynamically
        # did not seem to be working
        metadata = MetaData()
        metadata.schema = sf_schema
        # Made some edits to the langchains code to be able to pass the
        # query tag. Was successful and will contribute this to Langchain
        sql_db_chain = SQLDatabase(engine,metadata=metadata,query_tag="rajib-gen-ai")
        # Use the below for now
        # sql_db_chain = SQLDatabase()

        # SOURCE.YML
        source = f"""
                version: 2

                sources:
                    - name: source_01
                      description: This is a replica of the Snowflake database used by our app
                      database: pc_dbt_db
                      schema: dbt_rdeb
                      tables:
                          - name: customer
                            description: This the final customer table.
                          - name: stg_customer
                            description: the customer table for staging.
                          - name: stg_orders
                            description: One record per order. Includes cancelled and deleted orders."""
        # SCHEMA.YML
        schema = f"""
        version: 2

        models:
          - name: customer
            description: One record per customer
            columns:
              - name: customer_id
                description: Primary key
                tests:
                  - unique
                  - not_null
              - name: first_name
                description: The first name of the customer
              - name: last_name
                description: The last name of the customer
              - name: first_order_date
                description: NULL when a customer has not yet placed an order.
              - name: most_recent_order_date
                description: customers most recent date of order.
              - name: number_of_orders
                description: total number of orders by the customer
        
          - name: stg_customers
            description: This model cleans up customer data
            columns:
              - name: customer_id
                description: Primary key to identify a customer
                tests:
                  - unique
                  - not_null
              - name: first_name
                description: First name of the customer
              - name: last_name
                description: last name of the customer                
        
          - name: stg_orders
            description: This model cleans up order data
            columns:
              - name: order_id
                description: Primary key
                tests:
                  - unique
                  - not_null
              - name: customer_id
                description: Primary key to identify a customer
              - name: order_date
                description: date when customer placed the order.                
              - name: status
                tests:
                  - accepted_values:
                      values: ['placed', 'shipped', 'completed', 'return_pending', 'returned']"""

        # input = "Give me the name of all the customers who made a return in the last 10 years"
        # input = "How many orders have incorrect status"
        # input = "Show me all the delayed orders"
        # input = "Show me all the orders that are not delivered yet"
        # input = "Which of the open order deliveries have the maximum delay"
        input = "Show me the gap between the first and the recent order of a customer"
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)

        template = sf_prompt_template.format(schema=schema,source=source)
        # SQL_PROMPT = PromptTemplate(
        #     input_variables=["input","dialect"],
        #     template=sf_prompt_template
        # )
        SQL_PROMPT = PromptTemplate(
            input_variables=["input","dialect"],
            template=template
        )
        dbchain = SQLDatabaseChain.from_llm(llm,
                                            db=sql_db_chain,
                                            prompt=SQL_PROMPT,
                                            verbose=True,
                                            return_intermediate_steps=True)

        dbchain.top_k=100
        # resp = dbchain.run(input)
        resp = dbchain({"query":input})

        print("---Query----")
        print(resp['query'])
        print("---Result----")
        print(resp['result'])
        print("---intermediate_steps----")
        print(resp['intermediate_steps'])
        for steps in resp['intermediate_steps']:
            print(steps)


if __name__ == "__main__":
    sf = SnowflakeDBAssist()
    sf.get_response()
