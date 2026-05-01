import streamlit as st
import time
from PIL import Image
import io
import os

# SQLite Fix for ChromaDB on Windows
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from agents import PharmaAgentManager
from utils import PDFReporter

# Page Config
st.set_page_config(
    page_title="Pharma-Guard AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }

    .stButton>button {
        background: linear-gradient(45deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
    }

    h1, h2, h3 {
        color: #00d2ff !important;
        font-weight: 800 !important;
    }

    .status-log {
        font-family: 'Courier New', Courier, monospace;
        background: #1e1e1e;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00d2ff;
        font-size: 0.9em;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = PharmaAgentManager()

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/883/883356.png", width=100)
    st.title("Pharma-Guard AI")
    st.markdown("---")
    st.info("💡 **Gorev:** Vatandasin ilaci yanlis kullanmasini onlemek.")
    st.warning("⚠️ **Kritik Uyar:** Bilgi kaynagi %100 tibbi prospektuslerdir.")
    
    st.markdown("### Sistem Durumu")
    st.success("✅ Ready Mode")
    
    if st.button("Veritabanini Yenile"):
        with st.spinner("Indeksleme yapiliyor..."):
            st.session_state.agent_manager.rag.index_documents()
            st.success("Indeksleme tamamlandi!")

# Main UI
st.title("💊 Yapay Zeka Destekli Akilli Ilac Denetcisi")
st.markdown("""
    <p style='font-size: 1.2em; color: #a0a0a0;'>
    Görüntü işleme ve RAG teknolojisi ile prospektüs onaylı ilaç analizi.
    </p>
""", unsafe_allow_html=True)

# User Guide
with st.expander("📘 Kullanim Kilavuzu"):
    st.markdown("""
    1. **Veri Hazırlığı:** `data/corpus` klasörüne ilaçların PDF prospektüslerini yükleyin.
    2. **Giriş Yöntemi:** İlaç kutusunun fotoğrafını yükleyin veya ismini manuel girin.
    3. **Analiz:** 'Analizi Başlat' butonuna basın.
    4. **Rapor:** Sonuçları inceleyin ve profesyonel PDF raporunu indirin.
    ---
    *Not: Bu sistem bir yardımcıdır, tıbbi kararlar için her zaman bir doktora danışın.*
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📸 Gorsel Analiz")
    uploaded_file = st.file_uploader("Ilac kutusunun fotografini yukleyin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yuklenen Gorsel", use_column_width=True)
    
    st.markdown("---")
    st.subheader("✍️ Manuel Giris")
    manual_input = st.text_area("Ilac ismi ve detaylarini buraya yazin...")
    
    analyze_btn = st.button("Analizi Baslat")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if analyze_btn:
        if uploaded_file or manual_input:
            with st.status("Pharma-Guard Agents is basinda...", expanded=True) as status:
                start_time = time.time()
                
                # Input Handling
                if uploaded_file:
                    img_bytes = uploaded_file.getvalue()
                    mime_type = uploaded_file.type
                    results = st.session_state.agent_manager.orchestrate(img_bytes, is_image=True, mime_type=mime_type)
                else:
                    results = st.session_state.agent_manager.orchestrate(manual_input, is_image=False)
                
                # Logging
                for log in results["logs"]:
                    st.write(f"🔹 {log}")
                    time.sleep(0.5)
                
                st.session_state.analysis_results = results
                status.update(label=f"Analiz Tamamlandi! ({round(time.time()-start_time, 2)}s)", state="complete", expanded=False)
        else:
            st.error("Lutfen bir gorsel yukleyin veya metin girin.")

    if st.session_state.analysis_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📄 Analiz Raporu")
        st.markdown(st.session_state.analysis_results["report"])
        
        # Medical Disclaimer
        st.warning("⚠️ **ÖNEMLİ UYARI:** Bu rapor bilgilendirme amaçlıdır. İlacı kullanmadan önce mutlaka doktorunuza danışınız. Beklenmeyen bir etki görüldüğünde en yakın sağlık kuruluşuna başvurunuz.")
        
        # PDF Generation
        if st.button("PDF Raporu Indir"):
            def extract_section(text, section_num):
                import re
                patterns = [
                    r"1\.\s*(.*?)(?=\n2\.)",
                    r"2\.\s*(.*?)(?=\n3\.)",
                    r"3\.\s*(.*?)(?=\n4\.)",
                    r"4\.\s*(.*?)(?=\n5\.)",
                    r"5\.\s*(.*)$"
                ]
                try:
                    match = re.search(patterns[section_num-1], text, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                except:
                    pass
                return "Detaylar ana raporda mevcuttur."

            report_text = st.session_state.analysis_results["report"]
            report_data = {
                "summary": extract_section(report_text, 1),
                "indications": extract_section(report_text, 2),
                "warnings": extract_section(report_text, 3),
                "details": extract_section(report_text, 4),
                "sources": extract_section(report_text, 5),
                "disclaimer": "ONEMLI UYARI: Bu rapor bilgilendirme amaclidir. Ilaci kullanmadan once mutlaka doktorunuza danisiniz. Beklenmeyen bir etki goruldugunde en yakin saglik kurulusuna basvurunuz."
            }
            
            reporter = PDFReporter()
            reporter.generate_report(report_data, "pharma_report.pdf")
            
            with open("pharma_report.pdf", "rb") as f:
                st.download_button(
                    label="📥 Dosyayi Kaydet",
                    data=f,
                    file_name="PharmaGuard_Report.pdf",
                    mime="application/pdf"
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Analiz sonuclari burada goruntulenecektir.")

# Footer
st.markdown("---")
st.markdown("<center><small>Pharma-Guard AI © 2026 | Yapay Zeka Uygulamalari ve Veri Bilimi Dersi Projesi</small></center>", unsafe_allow_html=True)
