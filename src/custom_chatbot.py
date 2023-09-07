import os
import time
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

class CustomChatbot:
    """Custom Chatbot class to handle PDF documents and vector database."""
    
    def __init__(self, api_key: str, prompt_template: str, db_path: str = None,
                 chunk_size: int = 2000, chunk_overlap: int = 100):
        """Initialize CustomChatbot instance.

        Args:
            prompt_template: The prompt for the QA chain.
            db_path: Path to the directory where the vector database is stored.
            chunk_size: Size of chunks for text splitting. 
            chunk_overlap: Overlap between chunks for text splitting. 
        """
        os.environ['OPENAI_API_KEY'] = api_key
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chain = None
        self.directory = Path(db_path) / 'index_store' if db_path else Path('..') / 'index_store'


    def set_chain(self, chain):
        self.chain = chain

    def load_and_save_vector_db(self, documents: list, model_id: str, k_search: int):
        """Load and save a vector database and initialize a QA chain."""
        vectordb = self.create_vector_db_from_docs(documents)
        vectordb.save_local(self.directory)
        self.set_chain(self.create_qa_chain(vectordb, model_id, k_search))

    def create_vector_db_from_docs(self, documents: list):
        """Create a vector database from a list of documents."""
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return FAISS.from_documents(texts, OpenAIEmbeddings())

    def load_pdf_files(self, files: list):
        """Load PDF files and return their contents."""
        documents = []
        for pdf_path in files:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        return documents

    def create_qa_chain(self, vectordb, model_id: str, k_search: int):
        """Create a QA chain based on a vector database and a model identifier."""
        llm = ChatOpenAI(temperature=0, model_name=model_id)
        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": k_search}
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def upload_files(self, files: list, model_id: str, k_search: int):
        """Upload files and update the vector database."""
        documents = self.load_pdf_files([str(fn.name) for fn in files])
        self.load_and_save_vector_db(documents, model_id, k_search)
        return gr.Textbox.update(placeholder='Type your message here', lines=1, interactive=True)

    def get_vector_db(self, model_id: str, k_search: int):
        """Get the existing vector database and initialize a QA chain if existing."""
        if not (self.directory).exists():
            return gr.Textbox.update(
                placeholder='Document store does not exist, please upload document(s).',
                lines=1,
                interactive=False
            )
        vectordb = FAISS.load_local(self.directory, OpenAIEmbeddings())
        self.set_chain(self.create_qa_chain(vectordb, model_id, k_search))
        return gr.Textbox.update(placeholder='Type your message here', lines=1, interactive=True)

    @staticmethod
    def user(user_message: str, history: list):
        """Handle user messages."""
        return "", history + [[user_message, None]]

    @staticmethod
    def extract_source_and_page(documents: list):
        """Extract the source and page number from the source documents used to generate answers."""
        results = {}
        for document in documents:
            source = Path(document.metadata['source']).stem
            page = document.metadata['page'] + 1
            if source in results:
                results[source].append(page)
            else:
                results[source] = [page]
        return results

    def build_response_message(self, result: str, source_and_page: dict):
        """Build a response message with information about the source and page(s)."""
        response_message = f"{result}\n\nInformasjonen er hentet fra: \n"
        source_and_page_str = ''.join(
            [f'- Kilde: {source}, Side {sorted(pages)}\n' for source, pages in source_and_page.items()]
        )
        return f"{response_message} {source_and_page_str}"

    def generate_response(self, history: list):
        """Generate a response for the user based on their query and update the conversation history."""
        try:
            response = self.chain({"query": history[-1][0]})
            source_and_page = self.extract_source_and_page(response["source_documents"])
            response_message = self.build_response_message(response['result'], source_and_page)
        except Exception as e:
            response_message = (
                "Beklager, tekstens lengde overstiger modellens maksimale kapasitet for kontekstlengde."
                "Vennligst reduser k-search eller bytt til en modell med større kontekstlengde."
                )

        history[-1][1] = ""

        for char in response_message:
            history[-1][1] += char
            yield history
            time.sleep(0.01)