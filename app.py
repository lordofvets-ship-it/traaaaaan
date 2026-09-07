"""
🎙️ تطبيق تفريغ الصوت بالعامية المصرية
باستخدام نموذج Wav2Vec2 العربي
"""

import os
import tempfile
import time
import warnings
import locale
import io

# ============================================================
# 🛠️ إصلاح الترميز
# ============================================================
def fix_encoding():
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

fix_encoding()
warnings.filterwarnings("ignore")

# ============================================================
# 📦 استيراد المكتبات
# ============================================================
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# ============================================================
# 🎨 إعدادات الصفحة
# ============================================================
st.set_page_config(
    page_title="تفريغ الصوت - العامية المصرية",
    page_icon="🎙️",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.3rem;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 12px;
        border-right: 5px solid #FF4B4B;
        direction: rtl;
        font-size: 1.2rem;
        line-height: 2;
        font-family: 'Segoe UI', 'Tahoma', sans-serif;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #d93636;
        color: white;
    }
    .file-info {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🧠 تحميل النموذج
# ============================================================
@st.cache_resource
def load_model():
    """
    تحميل نموذج Wav2Vec2 العربي
    """
    with st.spinner("🔄 جاري تحميل نموذج التعرف على الصوت العربي... قد يستغرق 1-2 دقيقة"):
        try:
            processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53-arabic"
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-large-xlsr-53-arabic"
            )
            return processor, model
        except Exception as e:
            st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
            return None, None

# ============================================================
# 🎯 وظائف معالجة الصوت
# ============================================================
def convert_to_wav(audio_bytes, target_sr=16000):
    """
    تحويل الملف الصوتي إلى WAV
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت: {str(e)}")
        return None

def transcribe_audio(processor, model, audio_path):
    """
    تفريغ الملف الصوتي باستخدام Wav2Vec2
    """
    try:
        # تحميل الصوت
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # معالجة الصوت
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        
        # التفريغ
        with torch.no_grad():
            logits = model(input_values).logits
        
        # تحويل النتيجة لنص
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
    except Exception as e:
        st.error(f"❌ خطأ في التفريغ: {str(e)}")
        return None

# ============================================================
# 📱 واجهة التطبيق
# ============================================================

st.markdown('<h1 class="main-title">🎙️ تفريغ الصوت</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">مخصص للعامية المصرية | يعمل بالكامل في السحابة ☁️</p>', unsafe_allow_html=True)

# ============================================================
# 📂 رفع الملفات
# ============================================================
st.markdown("### 📤 اختر ملفاً صوتياً")

uploaded_file = st.file_uploader(
    "قم برفع الملف الصوتي هنا",
    type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
    help="الملفات المدعومة: WAV, MP3, M4A, FLAC, OGG, AAC",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.audio(uploaded_file, format="audio/wav")
    with col2:
        st.markdown(f"""
        <div class="file-info">
        📁 **الملف:** {uploaded_file.name}<br>
        📦 **الحجم:** {uploaded_file.size / 1024:.1f} كيلوبايت
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 🚀 زر التشغيل
# ============================================================
if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button("🎙️ ابدأ التفريغ", use_container_width=True)
    
    if start_button:
        # تحميل النموذج
        processor, model = load_model()
        if processor is None or model is None:
            st.stop()
        
        # تحويل الملف
        with st.spinner("🔄 جاري تحضير الملف للتفريغ..."):
            audio_bytes = uploaded_file.read()
            wav_path = convert_to_wav(audio_bytes)
            if wav_path is None:
                st.stop()
        
        # التفريغ
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 جاري تفريغ الصوت...")
        progress_bar.progress(30)
        
        start_time = time.time()
        text = transcribe_audio(processor, model, wav_path)
        end_time = time.time()
        
        # تنظيف الملف المؤقت
        try:
            os.unlink(wav_path)
        except:
            pass
        
        progress_bar.progress(100)
        status_text.text("✅ اكتمل التفريغ!")
        
        # عرض النتيجة
        if text:
            st.success(f"✅ تم التفريغ بنجاح في {end_time - start_time:.2f} ثانية")
            
            st.markdown("### 📝 النص المفهرس")
            st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
            
            # خيارات التحميل
            st.markdown("### 📥 خيارات التحميل")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.download_button(
                    label="📄 تحميل TXT",
                    data=text.encode('utf-8'),
                    file_name=f"transcript_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                srt_content = f"""1
00:00:00,000 --> 00:10:00,000
{text}"""
                st.download_button(
                    label="🎬 تحميل SRT",
                    data=srt_content.encode('utf-8'),
                    file_name=f"transcript_{int(time.time())}.srt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                if st.button("📋 نسخ النص", use_container_width=True):
                    st.code(text, language="text")
                    st.success("✅ تم النسخ!")
            
            with col4:
                words = len(text.split())
                st.metric("📊 الكلمات", words)
        else:
            st.error("❌ فشل التفريغ. حاول مرة أخرى.")

# ============================================================
# ℹ️ معلومات التطبيق
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ عن التطبيق")
    st.markdown("""
    **🎯 المميزات:**
    - ✅ دعم اللغة العربية
    - ✅ نموذج خفيف Wav2Vec2
    - ✅ يعمل في السحابة مجاناً
    - ✅ دعم صيغ متعددة
    
    **📌 التقنيات المستخدمة:**
    - Streamlit Cloud
    - Wav2Vec2 (Facebook)
    - Hugging Face Transformers
    
    **⏱️ وقت المعالجة:**
    - تحميل النموذج: 1-2 دقيقة (أول مرة فقط)
    - تفريغ الملف: حسب طول الملف
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 التثبيت المحلي")
    st.code("""
    pip install streamlit transformers torch librosa pydub
    streamlit run app.py
    """, language="bash")
    
    st.markdown("---")
    st.markdown("### 📚 صيغ مدعومة")
    st.markdown("""
    ✅ WAV
    ✅ MP3  
    ✅ M4A
    ✅ FLAC
    ✅ OGG
    ✅ AAC
    """)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
    🎙️ تطبيق تفريغ الصوت بالعامية المصرية | يعمل بالكامل على السحابة ☁️
    </div>
    """,
    unsafe_allow_html=True
)
