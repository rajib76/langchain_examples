import os

import kuzu
from dotenv import load_dotenv
from langchain.chains import KuzuQAChain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import KuzuGraph

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def get_db():
    db = kuzu.Database("../db/test_db")
    conn = kuzu.Connection(db)

    return conn,db


def insert_records(conn):
    conn.execute("CREATE NODE TABLE Movie (name STRING, PRIMARY KEY(name))")
    conn.execute("CREATE NODE TABLE Person (name STRING, birthDate STRING, PRIMARY KEY(name))")
    conn.execute("CREATE REL TABLE ActedIn (FROM Person TO Movie)")

    conn.execute("CREATE (:Person {name: 'Al Pacino', birthDate: '1940-04-25'})")
    conn.execute("CREATE (:Person {name: 'Robert De Niro', birthDate: '1943-08-17'})")
    conn.execute("CREATE (:Movie {name: 'The Godfather'})")
    conn.execute("CREATE (:Movie {name: 'The Godfather: Part II'})")
    conn.execute("CREATE (:Movie {name: 'The Godfather Coda: The Death of Michael Corleone'})")
    conn.execute(
        "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather' CREATE (p)-[:ActedIn]->(m)")
    conn.execute(
        "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)")
    conn.execute(
        "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather Coda: The Death of Michael Corleone' CREATE (p)-[:ActedIn]->(m)")
    conn.execute(
        "MATCH (p:Person), (m:Movie) WHERE p.name = 'Robert De Niro' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)")

if __name__=="__main__":
    conn,db = get_db()
    # insert_records(conn)
    graph = KuzuGraph(db)
    chain = KuzuQAChain.from_llm(
        ChatOpenAI(temperature=0), graph=graph, verbose=True
    )
    print(graph.get_schema)
    chain.run("Who played in The Godfather: Part II?")