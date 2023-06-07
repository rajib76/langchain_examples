from configparser import ConfigParser
from typing import Optional

import psycopg2
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool


class ExampleTool(BaseTool):
    name = "Database retrieval tool"
    description ="Useful when need to answer a question based on provided context"

    def run_tool(self):
        # database.ini file has below config
        # [pgdb]
        # host = localhost
        # database = postgres
        # user = xxx
        # password = xxx
        config_parser = ConfigParser()
        config_parser.read("./config/database.ini")
        db = {}
        params = config_parser.items("pgdb")
        for param in params:
            db[param[0]] = param[1]

        conn = psycopg2.connect(**db)
        cursor = conn.cursor()
        cursor.execute("""
        select id, name, salary from example
        """)

        results = cursor.fetchall()
        prev_context = ""
        context = ""
        for result in results:
            new_record = "Name:\n\n" + str(result[1]) + "Salary:\n\n" + str(result[2])+"\n\n"
            context = new_record + prev_context
            prev_context = context

        return context

    def _run(self,query:str,run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        return self.run_tool()

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("Does not support async")