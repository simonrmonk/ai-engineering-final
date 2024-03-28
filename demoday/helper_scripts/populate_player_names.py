from pathlib import Path

import pandas as pd
from langchain.sql_database import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone, PodSpec
from sqlalchemy import create_engine, text

pinecone_client = Pinecone()
# pinecone_client.create_index(
#     name="playernames",
#     dimension=1536,
#     metric="cosine",
#     spec=PodSpec(
#         environment="us-east-1-aws",
#     )
# )

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

engine = create_engine("sqlite:///fpl.db", echo=True)

HERE = Path(__file__).parents[1]

data_path = HERE.joinpath("FPL_csvs")

# Get a list of all CSV files in the directory
csv_files = [str(f) for f in data_path.glob("*.csv")]

csv_files = [x for x in csv_files if "cleaned_players" in x]

import logging

logging.error(f"Found these: {csv_files}")

table_names = list()
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    table_name = Path(csv_file).stem
    if table_name == "cleaned_players":
        df["full_name"] = df["first_name"] + " " + df["second_name"]
        df = df.drop(columns=["first_name", "second_name"])

    # Write the DataFrame to the SQLite database
    # Use the filename (without the .csv extension) as the table name
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    table_names.append(table_name)

db = SQLDatabase(engine, include_tables=["cleaned_players"])

# Define your SQL query
query = text("SELECT * FROM cleaned_players")

with engine.connect() as connection:
    # Execute the query
    result = connection.execute(query)

    # Fetch all rows from the result
    rows = result.fetchall()
    df = pd.DataFrame(rows)
    categorical_columns = df.select_dtypes(include=["object"]).columns

HERE = Path(__file__).parent
vector_store_path = str(HERE.joinpath("chroma_db_nouns"))

outputs = {}
categorical_columns = ["full_name"]
for category in categorical_columns:
    df_temp = df[[category]].drop_duplicates()
    items = df_temp[category].tolist()
    docs = list()
    for item in items:
        item = Document(page_content=item.strip())
        print(item.page_content)
        docs.append(item)

from langchain_pinecone import Pinecone

vector_store = Pinecone.from_documents(docs, embeddings_model, index_name="playernames")
