import os
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
import json
from utils import RagEngine
import time
from google.api_core import exceptions as google_exceptions

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
            self.model_name = "gemini-1.5-flash" 
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
        Sen, Gemini tabanlı, multimodal ve çoklu ajan ekosistemini yöneten baş mimarsın. 
        Görevin; ilaç analizini SIFIR HATA ve MAKSİMUM GÜVENİRLİK ile yapmaktır.

        ### KRİTİK OPERASYONEL KURALLAR:
        1. ÖZLÜLÜK: Raporu gereksiz tıbbi literatürle doldurma. Maddeler halinde, öz ve net bilgiler ver.
        2. KESİN FORMAT: Rapora asla giriş cümlesi (örn: 'İşte rapor...', '### RAPOR SONUCU ###') yazma. DOĞRUDAN '1. Ilac Kimlik Ozeti' başlığı ile başla.
        3. VERİ UYUŞMAZLIĞI: Görseldeki mg/dozaj ile prospektüsteki veri uyuşmuyorsa, bu uyumsuzluğu '3. Kritik Uyarilar ve Yan Etkiler' bölümünün en başına kocaman '!!! VERİ UYUŞMAZLIĞI TESPİT EDİLDİ !!!' yazarak belirt. Raporun yapısını bozma.
        4. GÜVEN PUANI: Her bilgi için (1-10) arası puan ver.
        5. HALÜSİNASYON ENGELİ: Prospektüste yazmayan hiçbir bilgiyi ekleme.
        """

        from PIL import Image
        import io
        
        prompt = """
        ### ROLE: VISION-SCANNER AGENT ###
        Görevin; ilaç kutusunun üzerindeki metinleri yüksek doğrulukla okumaktır (OCR).
        Aşağıdaki bilgileri JSON formatında çıkar:
        - Brand Name (Ticari Ad)
        - Active Ingredient (Etken Madde - Genelde küçük yazılır)
        - Dosage (Dozaj - örn: 500 mg, 15 ml)
        - Form (Tablet, Şurup, Ampul vb.)
        - Barcode (Varsa 13 haneli barkod numarası)
        
        KURAL: Yazı okunmuyorsa asla tahmin etme, 'UNREADABLE' değerini ata.
        Lütfen sadece JSON çıktısı ver.
        """
        
        try:
            # Prepare image for Gemini
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return f"Vision Error: Gorsel dosyasi acilamadi: {str(e)}"
            
        vision_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b"]
        last_error = ""
        
        for v_model in vision_models:
            try:
                client = genai.GenerativeModel(v_model)
                response = client.generate_content([prompt, img])
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = str(e)
                time.sleep(1) # Small delay between retries
                continue
                
        return f"Vision Error: Gorsel isleme motorlari yanit vermedi. Hata: {last_error}"

    def fast_summarize(self, raw_data):
        """Uses Groq (Llama-3-70B) for fast JSON structuring and summarization."""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are the Corporate-Analyst and Secretary. Convert raw drug data into a clean JSON and summarize findings quickly."},
                    {"role": "user", "content": f"Data: {raw_data}"}
                ],
                model="llama-3.1-70b-versatile",
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
            
            # Parse drug name for RAG
            try:
                # Clean JSON if wrapped in markdown
                clean_json = extracted_info.replace("```json", "").replace("```", "").strip()
                drug_data = json.loads(clean_json)
                drug_name_for_rag = f"{drug_data.get('Brand Name', '')} {drug_data.get('Active Ingredient', '')}".strip()
                if not drug_name_for_rag or drug_name_for_rag == "UNREADABLE":
                    drug_name_for_rag = extracted_info
            except:
                drug_name_for_rag = extracted_info
        else:
            logs.append("Text-Input: Metin isleniyor...")
            extracted_info = input_data
            drug_name_for_rag = input_data

        # 2. RAG Retrieval
        logs.append(f"RAG-Specialist: '{drug_name_for_rag}' icin veritabani taraniyor...")
        rag_context = self.analyze_with_rag(drug_name_for_rag)

        # 3. Final Synthesis (Master Orchestrator)
        logs.append("Master-Orchestrator: Rapor sentezleniyor...")
        
        final_prompt = f"""
        {self.master_prompt}
        
        GIRIS VERISI (Analiz edilecek ilac bilgileri):
        {extracted_info}
        
        RAG / PROSPEKTUS KAYNAK VERISI (Dogrulanmis tibbi bilgiler):
        {rag_context}
        
        ### RAPOR TALIMATI:
        Yukardaki verileri kullanarak profesyonel bir tibbi analiz raporu hazirla. 
        Her bölüm için mutlaka 1-10 arasi bir GUVEN PUANI (Confidence Score) belirt. 
        Eger prospektüs verisi ile giris verisi (mg, dozaj vb.) uyusmuyorsa 'VERI UYUSMAZLIGI' uyarisi ver.
        
        ### CIKTI HIYERARSISI (BU FORMATI KESINLIKLE KORU, ONCESINDE VEYA SONRASINDA HICBIR METIN YAZMA):
        1. Ilac Kimlik Ozeti
        [Icerik ve Guven Puani]
        
        2. Kullanim Amaci (Endikasyonlar)
        [Icerik ve Guven Puani]
        
        3. Kritik Uyarilar ve Yan Etkiler
        [Varsa Veri Uyusmazligi Uyarisi]
        [Icerik ve Guven Puani]
        
        4. Etken Madde ve Uretici Detaylari
        [Icerik ve Guven Puani]
        
        5. RAG / Kaynakca
        [Icerik]
        
        ---
        **ONEMLI UYARI:** Bu rapor bilgilendirme amaclidir. Ilaci kullanmadan once mutlaka doktorunuza danisiniz. Beklenmeyen bir etki goruldugunde en yakin saglik kurulusuna basvurunuz.
        """
        
        try:
            # Groq model fallback list to ensure maximum reliability
            messages = [
                {"role": "system", "content": "Sen Pharma-Guard sisteminin bashekimisin. Turkce, kesin, dogru ve kisa tipbi analiz raporlari uretirsin."},
                {"role": "user", "content": final_prompt}
            ]
            
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=2048
                )
            except:
                # Fallback to faster/lighter model if 70B fails (rate limit, decommission, etc.)
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model="llama-3.1-8b-instant",
                    temperature=0.2,
                    max_tokens=2048
                )
                
            report_text = chat_completion.choices[0].message.content
        except Exception as e:
            report_text = f"Sentez Hatasi (Groq Llama-3): {str(e)}"
        
        return {
            "report": report_text,
            "logs": logs,
            "raw_info": extracted_info
        }
