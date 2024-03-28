from pathlib import Path
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone, PodSpec

HERE = Path(__file__).parent

pinecone_client = Pinecone()

pinecone_client.create_index(
    name="unstructurednew",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east-1-aws",
    )
)


llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_link = HERE.joinpath("journal.pone.0246698.pdf")
loader = PyMuPDFLoader(
    str(pdf_link),
)

documents = loader.load()

true_link = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0246698"

for doc in documents:
    doc.metadata["source"] = true_link

chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

research_chunks = text_splitter.split_documents(documents)

docs = list()
from langchain_core.documents import Document

filter_links_list = []

chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

links = [
    "https://www.fantasyfootballscout.co.uk/2024/03/18/fpl-gameweek-30-early-scout-picks-chelsea-double-up"
]

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

for link in links:
    if link in filter_links_list:
        continue
    print(link)

    html_header_splits = html_splitter.split_text_from_url(link)
    toss = html_header_splits.pop(0)
    html_header_splits = html_header_splits[:5]
    metadata = dict()
    metadata["source"] = link

    splits = text_splitter.split_documents(html_header_splits)

    for chunk in splits:
        text = list()
        if len(chunk.page_content) < 80:
            continue
        text.append(chunk.page_content)

        for _, value in chunk.metadata.items():
            text.append(value)

        text = " ".join(text)

        chunk = Document(page_content=text, metadata=metadata.copy())
        docs.append(chunk)

final_docs = docs + research_chunks

vector_store_path = str(HERE.joinpath("chroma_db_unstructured"))

from langchain_pinecone import Pinecone

# vectorstore = Chroma.from_documents(
#     docs, embeddings_model, persist_directory=vector_store_path
# )
vector_store = Pinecone.from_documents(final_docs, embeddings_model, index_name="unstructurednew")
