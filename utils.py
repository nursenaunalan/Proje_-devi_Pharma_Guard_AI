import os
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
            try:
                results = self.vectorstore.similarity_search(text, k=k)
                if not results:
                    return "ILGILI PROSPEKTUS BULUNAMADI. Lutfen genel tibbi bilgilerle degil, prospektus eksikligi uyarisiyla cevap ver."
                return "\n\n-- KAYNAK KESITI --\n" + "\n".join([doc.page_content for doc in results])
            except Exception as e:
                return f"Arama Hatasi: {str(e)}"
        return "Veritabani yuklenemedi veya bos."

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
        
        # Header Styling
        self.set_text_color(58, 123, 213) # Blue accent
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, self.clean_text("PHARMA-GUARD AI ANALIZ RAPORU"), ln=True, align="C")
        self.set_draw_color(58, 123, 213)
        self.line(10, 25, 200, 25)
        self.ln(10)

        sections = [
            ("1. Ilac Kimlik Ozeti", "summary", (0, 0, 0)),
            ("2. Kullanim Amaci (Endikasyonlar)", "indications", (0, 0, 0)),
            ("3. Kritik Uyarilar ve Yan Etkiler", "warnings", (255, 0, 0)), # Red for warnings
            ("4. Etken Madde ve Uretici Detaylari", "details", (0, 0, 0)),
            ("5. RAG / Kaynakca", "sources", (100, 100, 100)) # Grey for sources
        ]

        for title, key, color in sections:
            self.set_text_color(*color)
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 10, self.clean_text(title), ln=True)
            
            self.set_text_color(0, 0, 0)
            self.set_font("Helvetica", size=10)
            content = data.get(key, "Detayli bilgi bulunamadi.")
            self.multi_cell(0, 7, self.clean_text(content))
            self.ln(5)
            # Add a subtle line between sections
            self.set_draw_color(230, 230, 230)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)
        
        # Medical Disclaimer at the end
        self.ln(5)
        self.set_fill_color(255, 243, 205) # Light yellow background
        self.set_text_color(133, 100, 4) # Dark brownish yellow
        self.set_font("Helvetica", "B", 9)
        disclaimer = data.get("disclaimer", "")
        self.multi_cell(0, 7, self.clean_text(disclaimer), border=1, fill=True)
        
        self.output(filename)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
