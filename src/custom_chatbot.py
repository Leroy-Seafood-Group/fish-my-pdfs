import time
from pathlib import Path
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class CustomChatbot:
    """Custom Chatbot class to handle PDF documents and vector database."""
    
    def __init__(self, prompt_template: str, custom_files):
        """Initialize CustomChatbot instance.

        Args:
            prompt_template: The prompt for the QA chain.
            custom_files: object to handle PDF documents and vector database.
        """
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.custom_files = custom_files
        self.chain = None

    def set_chain(self, chain):
        self.chain = chain

    def load_and_save_vector_db(self, documents: list, model_id: str, k_search: int):
        """Load and save a vector database and initialize a QA chain."""
        vectordb = self.custom_files.create_vector_db_from_docs(documents)
        self.set_chain(self.create_qa_chain(vectordb, model_id, k_search))

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
        documents = self.custom_files.load_pdf_files([str(fn.name) for fn in files])
        self.load_and_save_vector_db(documents, model_id, k_search)
        return gr.Textbox.update(placeholder='Type your message here', lines=1, interactive=True)

    def get_vector_db(self, model_id: str, k_search: int):
        """Get the existing vector database and initialize a QA chain if existing."""
        placeholder_text = 'Document store does not exist, please upload document(s).'
        interactive = False
        vector_db = self.custom_files.load_local_db()
        if vector_db: 
            self.set_chain(self.create_qa_chain(vector_db, model_id, k_search))
            placeholder_text = 'Type your message here'
            interactive = True
            
        return gr.Textbox.update(placeholder=placeholder_text, lines=1, interactive=interactive)

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