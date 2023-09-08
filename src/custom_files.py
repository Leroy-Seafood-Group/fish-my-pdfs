from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class CustomFiles: 

    def __init__(self, chunk_size: int = 2000, chunck_overlap: int = 100, db_path: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunck_overlap
        self.directory = Path(db_path) / 'index_store' if db_path else Path('..') / 'index_store'

    def create_vector_db_from_docs(self, documents: list):
        """Create a vector database from a list of documents."""
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        vectordb =  FAISS.from_documents(texts, OpenAIEmbeddings())
        vectordb.save_local(self.directory)
        return vectordb

    def load_pdf_files(self, files: list):
        """Load PDF files and return their contents."""
        documents = []
        for pdf_path in files:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        return documents

    def load_local_db(self):
        """Load the existing vector database if existing."""
        if self.directory.exists():
            return FAISS.load_local(self.directory, OpenAIEmbeddings())

        return None