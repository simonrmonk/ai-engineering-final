from pathlib import Path

import pandas as pd
from langchain.sql_database import SQLDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from sqlalchemy import create_engine

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
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    table_names.append(table_name)

db = SQLDatabase(engine, include_tables=["cleaned_players"])

vectorstore = PineconeVectorStore(embedding=embeddings_model, index_name="playernames")

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.75}
)
retrievers = dict()
retrievers["full_name"] = retriever
