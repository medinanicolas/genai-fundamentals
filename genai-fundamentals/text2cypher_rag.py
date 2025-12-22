import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from gemini_llm import GeminiLLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever


# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create LLM 
#t2c_llm = OpenAILLM(model_name="gpt-5-mini", model_params={"reasoning_effort": "high"})
t2c_llm = GeminiLLM(model_name="gemini-3-flash-preview")

# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
]

# Specify your own Neo4j schema
neo4j_schema = """
Node properties:
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Genre {name: STRING}
User {name: STRING}

Relationship properties:
ACTED_IN {role: STRING}
RATED {rating: INTEGER}

The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:User)-[:RATED]->(:Movie)
(:Movie)-[:IN_GENRE]->(:Genre)
"""

# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    examples=examples,
    neo4j_schema=neo4j_schema,
)

llm = GeminiLLM(model_name="gemini-3-flash-preview")

rag = GraphRAG(retriever=retriever, llm=llm)

query_text = "Which movies did Hugo Weaving star in?"
response = rag.search(
    query_text=query_text,
    return_context=True
    )
print("USER INPUT:", query_text)
print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)
print("----------------")

# for query_text in ["Which movies did Hugo Weaving star in?", "What is the highest rating for Goodfellas?", "What is the average user rating for the movie Toy Story?", "What user gives the lowest ratings?"]:
#     response = rag.search(
#         query_text=query_text,
#         return_context=True
#         )
#     print("USER INPUT:", query_text)
#     print(response.answer)
#     print("CYPHER :", response.retriever_result.metadata["cypher"])
#     print("CONTEXT:", response.retriever_result.items)
#     print("----------------")

driver.close()
