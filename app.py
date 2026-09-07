"""
🎙️ تطبيق تفريغ الصوت بالعامية المصرية
مصمم للعمل على Streamlit Cloud
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
from pydub import AudioSegment
from faster_whisper import WhisperModel

# ============================================================
# 🎨 إعدادات الصفحة
# ============================================================
st.set_page_config(
    page_title="تفريغ الصوت - العامية المصرية",
    page_icon="🎙️",
    layout="wide"
)

# CSS مخصص
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
# 🧠 تحميل النموذج (مع Cache)
# ============================================================
@st.cache_resource
def load_model():
    """
    تحميل نموذج Faster-Whisper المدرب على العامية المصرية
    """
    with st.spinner("🔄 جاري تحميل النموذج... قد يستغرق 1-2 دقيقة"):
        try:
            # استخدام النموذج الخفيف المدرب على المصرية
            model = WhisperModel(
                'Mano200600/faster-whisper-small-egyptian-ar',
                device='cpu',
                compute_type='int8'
            )
            st.success("✅ تم تحميل النموذج بنجاح!")
            return model
        except Exception as e:
            st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
            return None

# ============================================================
# 🎯 وظائف معالجة الصوت
# ============================================================
def convert_to_wav(audio_bytes, target_sr=16000):
    """
    تحويل الملف الصوتي إلى WAV بمعدل 16 كيلو هرتز
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # ضبط الإعدادات
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # حفظ في ملف مؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"❌ خطأ في تحويل الصوت: {str(e)}")
        return None

def transcribe_audio(model, audio_path):
    """
    تفريغ الملف الصوتي باستخدام Faster-Whisper
    """
    try:
        segments, info = model.transcribe(audio_path, language="ar")
        
        # جمع النص من جميع الأجزاء
        full_text = " ".join([segment.text for segment in segments])
        return full_text
        
    except Exception as e:
        st.error(f"❌ خطأ في التفريغ: {str(e)}")
        return None

# ============================================================
# 📱 واجهة التطبيق
# ============================================================

# العنوان الرئيسي
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
    # عرض معلومات الملف
    col1, col2, col3 = st.columns([2, 1, 1])
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
        # ============================================================
        # 1. تحميل النموذج
        # ============================================================
        model = load_model()
        if model is None:
            st.stop()
        
        # ============================================================
        # 2. تحويل الملف
        # ============================================================
        with st.spinner("🔄 جاري تحضير الملف للتفريغ..."):
            audio_bytes = uploaded_file.read()
            wav_path = convert_to_wav(audio_bytes)
            if wav_path is None:
                st.stop()
        
        # ============================================================
        # 3. التفريغ مع مؤشر التقدم
        # ============================================================
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 جاري تفريغ الصوت...")
        progress_bar.progress(20)
        
        start_time = time.time()
        text = transcribe_audio(model, wav_path)
        end_time = time.time()
        
        # تنظيف الملف المؤقت
        try:
            os.unlink(wav_path)
        except:
            pass
        
        progress_bar.progress(100)
        status_text.text("✅ اكتمل التفريغ!")
        
        # ============================================================
        # 4. عرض النتيجة
        # ============================================================
        if text:
            st.success(f"✅ تم التفريغ بنجاح في {end_time - start_time:.2f} ثانية")
            
            # عرض النص
            st.markdown("### 📝 النص المفهرس")
            st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
            
            # ============================================================
            # 5. خيارات التحميل
            # ============================================================
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
                # تحميل كـ SRT
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
                # إحصائيات
                words = len(text.split())
                chars = len(text)
                st.metric("📊 الكلمات", words)
        else:
            st.error("❌ فشل التفريغ. حاول مرة أخرى أو جرب ملفاً آخر.")

# ============================================================
# ℹ️ معلومات التطبيق (في الـ Sidebar)
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ عن التطبيق")
    st.markdown("""
    **🎯 المميزات:**
    - ✅ دعم العامية المصرية
    - ✅ نموذج خفيف وسريع
    - ✅ يعمل في السحابة مجاناً
    - ✅ دعم صيغ متعددة
    - ✅ تحميل النص بأكثر من صيغة
    
    **📌 التقنيات المستخدمة:**
    - Streamlit Cloud
    - Faster-Whisper
    - نموذج `faster-whisper-small-egyptian-ar`
    
    **⏱️ وقت المعالجة:**
    - تحميل النموذج: 1-2 دقيقة (أول مرة فقط)
    - تفريغ الملف: حسب طول الملف
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔧 التثبيت المحلي")
    st.code("""
    pip install streamlit faster-whisper pydub
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

# ============================================================
# 🏁 نهاية التطبيق
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
    🎙️ تطبيق تفريغ الصوت بالعامية المصرية | يعمل بالكامل على السحابة ☁️
    </div>
    """,
    unsafe_allow_html=True
)
