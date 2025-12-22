import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
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
t2c_llm = OpenAILLM(model_name="gpt-5-mini", model_params={"reasoning_effort": "high"})

# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating"
]

# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    examples=examples,
)

llm = OpenAILLM(model_name="gpt-5-mini", model_params={"reasoning_effort": "high"})

rag = GraphRAG(retriever=retriever, llm=llm)

#query_text = "Which movies did Hugo Weaving star in?"

for query_text in ["Which movies did Hugo Weaving star in?", "What is the highest rating for Goodfellas?", "What is the average user rating for the movie Toy Story?", "What user gives the lowest ratings?"]:
    response = rag.search(
        query_text=query_text,
        return_context=True
        )
    print("USER INPUT:", query_text)
    print(response.answer)
    print("CYPHER :", response.retriever_result.metadata["cypher"])
    print("CONTEXT:", response.retriever_result.items)
    print("----------------")

driver.close()
