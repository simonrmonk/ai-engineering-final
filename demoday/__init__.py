from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
new_line_character = "\n"
