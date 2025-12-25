from dotenv import load_dotenv
import os
from src.helper import *
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_doc = load_pdf_files("data/")
new_doc = filter_for_extracted_doc(extracted_doc)
text_chunk = text_splitter(new_doc)

embedding = downlad_embedding()

pinecone_api_key = PINECONE_API_KEY
pinecone = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pinecone.has_index(index_name):
    pinecone.create_index(
        name=index_name,
        dimension=384,
        spec=ServerlessSpec(cloud="aws",region="us-east-1"),
        metric="cosine"
    )

index= pinecone.Index(name=index_name)


doc_search = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name= index_name
    )