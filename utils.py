import os
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PIL import Image
import io
import base64

class RagEngine:
    def __init__(self, corpus_path="data/corpus", db_path="data/chroma_db"):
        self.corpus_path = corpus_path
        self.db_path = db_path
        
        # Ensure directories exist
        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path)
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        
        try:
            if os.listdir(self.db_path):
                self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            else:
                self.index_documents()
        except Exception as e:
            print(f"ChromaDB Baslatma Hatasi: {e}")
            self.index_documents()

    def index_documents(self):
        documents = []
        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path)
            
        for file in os.listdir(self.corpus_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.corpus_path, file))
                documents.extend(loader.load())
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings, 
                persist_directory=self.db_path
            )
        else:
            print("No PDF documents found in corpus.")

    def query(self, text, k=3):
        if self.vectorstore:
            results = self.vectorstore.similarity_search(text, k=k)
            return "\n".join([doc.page_content for doc in results])
        return "No database initialized."

class PDFReporter(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "Pharma-Guard AI Analysis Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def clean_text(self, text):
        """Replaces Turkish characters with their Latin equivalents for FPDF compatibility."""
        replacements = {
            'ğ': 'g', 'Ğ': 'G',
            'ü': 'u', 'Ü': 'U',
            'ş': 's', 'Ş': 'S',
            'ı': 'i', 'İ': 'I',
            'ö': 'o', 'Ö': 'O',
            'ç': 'c', 'Ç': 'C'
        }
        for tr, lat in replacements.items():
            text = text.replace(tr, lat)
        return text

    def generate_report(self, data, filename="report.pdf"):
        self.add_page()
        
        # Title
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.clean_text("1. Ilac Kimlik Ozeti"), ln=True)
        self.set_font("Helvetica", size=12)
        self.multi_cell(0, 10, self.clean_text(data.get("summary", "N/A")))
        self.ln(5)
        
        # Indications
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.clean_text("2. Kullanim Amaci (Endikasyonlar)"), ln=True)
        self.set_font("Helvetica", size=12)
        self.multi_cell(0, 10, self.clean_text(data.get("indications", "N/A")))
        self.ln(5)
        
        # Warnings
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 0, 0)
        self.cell(0, 10, self.clean_text("3. Kritik Uyarilar ve Yan Etkiler"), ln=True)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", size=12)
        self.multi_cell(0, 10, self.clean_text(data.get("warnings", "N/A")))
        self.ln(5)
        
        # Details
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.clean_text("4. Etken Madde ve Uretici Detaylari"), ln=True)
        self.set_font("Helvetica", size=12)
        self.multi_cell(0, 10, self.clean_text(data.get("details", "N/A")))
        self.ln(5)
        
        # RAG Sources
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.clean_text("5. RAG / Kaynakca"), ln=True)
        self.set_font("Helvetica", size=12)
        self.multi_cell(0, 10, self.clean_text(data.get("sources", "N/A")))
        
        self.output(filename)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
