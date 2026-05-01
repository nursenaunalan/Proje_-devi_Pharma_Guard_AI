import os
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
import json
from utils import RagEngine
import time

load_dotenv()

class PharmaAgentManager:
    def __init__(self):
        # API Keys - Check environment first, then st.secrets (for deployment)
        import streamlit as st
        self.gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        
        if not self.gemini_key or "your_gemini" in self.gemini_key:
            st.error("GEMINI_API_KEY bulunamadi! Lutfen .env dosyasini veya Streamlit Secrets ayarlarini kontrol edin.")
            st.stop()
        
        if not self.groq_key or "your_groq" in self.groq_key:
            st.error("GROQ_API_KEY bulunamadi! Lutfen .env dosyasini veya Streamlit Secrets ayarlarini kontrol edin.")
            st.stop()

        # Clients
        try:
            genai.configure(api_key=self.gemini_key)
            # Use 'latest' suffix for better compatibility with different API versions
            self.model_name = "gemini-1.5-flash-latest" 
            self.gemini_model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            st.error(f"Gemini Baglantisi Kurulamadi: {str(e)}")
            st.stop()

        try:
            self.groq_client = Groq(api_key=self.groq_key)
        except Exception as e:
            st.error(f"Groq Baglantisi Kurulamadi: {str(e)}")
            st.stop()
        
        # RAG Engine
        try:
            self.rag = RagEngine()
        except Exception as e:
            print(f"RAG Baslatma Hatasi (PDF eklemeyi unutmus olabilirsiniz): {str(e)}")
            self.rag = None

        # Master Prompt
        self.master_prompt = """
        ### ROLE: PHARMA-GUARD MASTER ORCHESTRATOR (PG-MO) ###
        Sen, Gemini 2.0 tabanlı, multimodal yeteneklere sahip ve çoklu ajan (Multi-Agent) ekosistemini yöneten baş mimarsın. 
        Görevin; görsel veya metinsel girişi alınan bir ilacı, sıfır hata toleransı ile analiz etmektir.

        ### OPERASYONEL PROTOKOLLER VE KISITLAMALAR:
        - GÜVEN PUANI (Confidence Score): Her bilgi parçası için 1-10 arası bir puan ver. 
        - HALÜSİNASYON ENGELİ: Eğer ilacın etken maddesi ile prospektüs bilgisi eşleşmiyorsa, 'Fact-Checker' devreye girsin ve süreci durdurup hata mesajı versin.
        - DİL VE ÜSLUP: Rapor tamamen Türkçe, tıbbi terimleri parantez içinde açıklayan, güven veren ve profesyonel bir tonda olmalıdır.
        - KURAL: Yazı okunmuyorsa asla tahmin etme!
        - KURAL: Bilgiler arasında 1 mg fark olsa bile raporu blokla ve 'VERİ UYUŞMAZLIĞI' alarmı ver.
        """

    def vision_scan(self, image_bytes, mime_type="image/jpeg"):
        """Uses Gemini 2.0 Flash for vision analysis."""
        try:
            prompt = """
            Analyze this drug box image as the [Vision-Scanner] agent. 
            Extract: Brand Name (Ticari Ad), Active Ingredient (Etken Madde), Dosage (mg/ml), Form (Tablet/Syrup), and Barcode. 
            If the text is unreadable, respond with 'UNREADABLE'.
            Output MUST be in valid JSON format.
            """
            response = self.gemini_model.generate_content([
                prompt, 
                {"mime_type": mime_type, "data": image_bytes}
            ])
            return response.text
        except Exception as e:
            return f"Vision Error: {str(e)}"

    def fast_summarize(self, raw_data):
        """Uses Groq (Llama-3-70B) for fast JSON structuring and summarization."""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are the Corporate-Analyst and Secretary. Convert raw drug data into a clean JSON and summarize findings quickly."},
                    {"role": "user", "content": f"Data: {raw_data}"}
                ],
                model="llama3-70b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Groq Error: {str(e)}"

    def analyze_with_rag(self, drug_name):
        """Uses RAG to get technical info from PDFs."""
        if self.rag:
            context = self.rag.query(drug_name)
            return context
        return "RAG sistemi aktif degil. (data/corpus klasorunde PDF bulunamadi)"

    def orchestrate(self, input_data, is_image=False, mime_type="image/jpeg"):
        """Main orchestration logic."""
        logs = []
        
        # 1. Vision Scan or Initial Text Processing
        if is_image:
            logs.append("Vision-Scanner: Gorsel analiz ediliyor...")
            extracted_info = self.vision_scan(input_data, mime_type=mime_type)
        else:
            logs.append("Text-Input: Metin isleniyor...")
            extracted_info = input_data

        # 2. RAG Retrieval
        logs.append("RAG-Specialist: Yerel veritabani taraniyor...")
        rag_context = self.analyze_with_rag(extracted_info)

        # 3. Final Synthesis (Master Orchestrator)
        logs.append("Master-Orchestrator: Rapor sentezleniyor...")
        
        final_prompt = f"""
        {self.master_prompt}
        
        GIRIS VERISI (Extracted):
        {extracted_info}
        
        RAG KAYNAK VERISI:
        {rag_context}
        
        Lutfen asagidaki hiyerarside raporu olustur:
        1. Ilac Kimlik Ozeti
        2. Kullanim Amaci (Endikasyonlar)
        3. Kritik Uyarilar ve Yan Etkiler
        4. Etken Madde ve Uretici Detaylari
        5. RAG / Kaynakca
        """
        
        try:
            response = self.gemini_model.generate_content(final_prompt)
            report_text = response.text
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                logs.append("⚠️ Kota sinirina takildi, 5 saniye bekleniyor...")
                time.sleep(5)
                response = self.gemini_model.generate_content(final_prompt)
                report_text = response.text
            else:
                report_text = f"Sentez Hatasi: {str(e)}"
        
        return {
            "report": report_text,
            "logs": logs,
            "raw_info": extracted_info
        }
