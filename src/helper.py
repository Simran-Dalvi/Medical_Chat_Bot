from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Extracting pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    document = loader.load()
    return document


def filter_for_extracted_doc(docs:List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                metadata={'source':src},
                page_content=doc.page_content
            )
        )

    return minimal_docs

# now split documents into smaller chunks
def text_splitter(doc):
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap =20,
        length_function = len
    )
    text = spliter.split_documents(doc)
    return text

# hugging face for embedding
def downlad_embedding():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embed = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embed
