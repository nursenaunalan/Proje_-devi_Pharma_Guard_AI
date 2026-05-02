# 💊 Pharma-Guard AI : Yapay Zeka Destekli Akıllı İlaç Denetçisi

Bu proje, görüntü işleme (Computer Vision) ve doğal dil işleme (NLP) teknolojilerini birleştirerek toplum sağlığına katkı sunmak amacıyla geliştirilmiş otonom bir **Çoklu Ajan Sistemidir (Multi-Agent System)**.

Sistem sadece basit bir bilgi arama motoru değil; **RAG (Retrieval-Augmented Generation)** teknolojisini kullanarak tıbbi prospektüslerden gerçek zamanlı doğrulama yapan ve kendi verisini denetleyen profesyonel bir asistan mimarisidir.

## ✨ Temel Özellikler

- 👁️ **Görsel Analiz (OCR):** İlaç kutusunun fotoğrafından marka, etken madde ve dozaj bilgilerini **Gemini 1.5** vizyon modelleriyle yüksek doğrulukla çıkarır.
- 📚 **RAG Tabanlı Doğrulama:** Analiz edilen ilacı, yerel veritabanındaki (ChromaDB) PDF prospektüsleriyle eşleştirir ve doğrular.
- 🧠 **Multi-Agent Orkestrasyonu:** Metin sentezleme ve analiz işlemleri için **Groq Llama-3 (70B)** modeli kullanılarak Google API kota limitleri aşılır ve inanılmaz bir hız elde edilir.
- 🚨 **Kritik Veri Uyuşmazlığı Alarmı (Fact-Checking):** Görseldeki dozaj bilgisi ile prospektüsteki bilgi arasında 0.1 mg bile fark varsa, sistem anında görsel bir "Kritik Uyarı" verir.
- 📄 **Profesyonel PDF Raporlama:** Analiz sonuçlarını yapılandırılmış, renk kodlamalı (uyarılar kırmızı) ve temiz bir PDF belgesi olarak indirmenizi sağlar.

## 🛠️ Kullanılan Teknolojiler

- **Frontend:** Streamlit
- **AI Modelleri:** Google Gemini 1.5 Flash/Pro (Görsel İşleme), Groq Llama-3.1-70B (Metin Sentezleme ve Orkestrasyon)
- **Vektör Veritabanı & RAG:** LangChain, ChromaDB, HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **Raporlama:** FPDF2

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
Projeyi yerel makinenizde çalıştırmak için Python 3.9+ gereklidir.

### 2. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

### 3. Çevresel Değişkenler (.env)
Proje dizininde bir `.env` dosyası oluşturun ve aşağıdaki API anahtarlarını ekleyin:
```env
GEMINI_API_KEY=sizin_gemini_api_anahtariniz
GROQ_API_KEY=sizin_groq_api_anahtariniz
```
*(Eğer Streamlit Cloud üzerinde yayınlıyorsanız, bu anahtarları `Advanced Settings -> Secrets` bölümüne eklemelisiniz.)*

### 4. Veri Hazırlığı
`data/corpus/` klasörünün içine ilaçlara ait PDF formatındaki prospektüsleri kopyalayın. Sistem ilk açılışta bu dosyaları okuyup ChromaDB vektör veritabanına kaydedecektir.

### 5. Uygulamayı Başlatma
```bash
streamlit run app.py
```

## 📘 Kullanım Kılavuzu

1. **Giriş Yöntemi Seçin:** İster ilaç kutusunun net bir fotoğrafını yükleyin, isterseniz de ilacın adını manuel olarak metin kutusuna girin.
2. **Analizi Başlatın:** Butona tıkladığınızda ajanlar sırasıyla devreye girer (OCR -> RAG -> Llama-3 Sentezi).
3. **Sonuçları İnceleyin:** Ekranda beliren 5 maddelik detaylı raporu ve varsa kırmızı renkli uyuşmazlık uyarılarını inceleyin.
4. **PDF İndirin:** Sağlanan buton ile bu raporu bilgisayarınıza PDF olarak kaydedebilirsiniz.

## ⚠️ Yasal Uyarı
**ÖNEMLİ:** Bu yazılım tamamen eğitim ve bilgilendirme amaçlıdır. Herhangi bir ilacı kullanmadan önce mutlaka doktorunuza danışınız. Beklenmeyen bir etki görüldüğünde en yakın sağlık kuruluşuna başvurunuz.
