import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from gtts import gTTS
import os
import base64
from io import BytesIO
import PyPDF2
import docx
import tempfile

# Page Configuration
st.set_page_config(
    page_title="Smart Document Summarization",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        color: #000000 !important;
    }
    .summary-box p {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'max_length' not in st.session_state:
    st.session_state.max_length = 150
if 'min_length' not in st.session_state:
    st.session_state.min_length = 50

@st.cache_resource
def load_model():
    """Load T5 model and tokenizer"""
    try:
        model_path = "t5-small
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return None

def generate_summary(text, tokenizer, model, max_length=150, min_length=50):
    """Generate summary using T5 model"""
    try:
        # Preprocess text
        text = "summarize: " + text
        
        # Tokenize
        inputs = tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def text_to_speech(text, lang='en'):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Document Summarization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your document, get a summary, and listen to it in audio!</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Model could not be loaded. Please check the model path.")
        return
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Update session state when sliders change
    new_max_length = st.sidebar.slider(
        "Summary Maximum Length", 
        50, 300, 
        st.session_state.max_length, 
        10,
        key="max_slider"
    )
    
    new_min_length = st.sidebar.slider(
        "Summary Minimum Length", 
        20, 100, 
        st.session_state.min_length, 
        10,
        key="min_slider"
    )
    
    # Ensure min_length is less than max_length
    if new_min_length >= new_max_length:
        st.sidebar.warning("⚠️ Minimum length must be less than maximum length!")
        new_min_length = new_max_length - 10
    
    # Update session state
    st.session_state.max_length = new_max_length
    st.session_state.min_length = new_min_length
    
    audio_lang = st.sidebar.selectbox(
        "Audio Language",
        ["en", "ur", "hi"],
        format_func=lambda x: {"en": "English", "ur": "Urdu", "hi": "Hindi"}[x]
    )
    
    st.sidebar.info(f"Current Settings:\n- Max: {st.session_state.max_length}\n- Min: {st.session_state.min_length}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Text Input", "File Upload"]
        )
        
        input_text = ""
        
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter your text here:",
                height=300,
                placeholder="Paste your document text here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload document (PDF, DOCX, TXT)",
                type=['pdf', 'docx', 'txt']
            )
            
            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                with st.spinner("Reading file..."):
                    if file_type == 'pdf':
                        input_text = extract_text_from_pdf(uploaded_file)
                    elif file_type == 'docx':
                        input_text = extract_text_from_docx(uploaded_file)
                    elif file_type == 'txt':
                        input_text = uploaded_file.read().decode('utf-8')
                
                if input_text:
                    st.success(f"✅ File successfully read! ({len(input_text)} characters)")
                    with st.expander("Extracted Text Preview"):
                        st.text(input_text[:500] + "..." if len(input_text) > 500 else input_text)
        
        # Generate summary button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if input_text and len(input_text.strip()) > 50:
                with st.spinner("Generating summary..."):
                    summary = generate_summary(
                        input_text,
                        tokenizer,
                        model,
                        st.session_state.max_length,
                        st.session_state.min_length
                    )
                    
                    if summary:
                        st.session_state.summary = summary
                        
                        # Generate audio
                        with st.spinner("Generating audio..."):
                            audio_bytes = text_to_speech(summary, audio_lang)
                            if audio_bytes:
                                st.session_state.audio_bytes = audio_bytes
                        
                        st.success("✅ Summary and audio generated successfully!")
            else:
                st.warning("⚠️ Please enter at least 50 characters of text.")
    
    with col2:
        st.header("📊 Output")
        
        if st.session_state.summary:
            # Display summary
            st.subheader("📋 Summary")
            st.markdown(
                f'<div class="summary-box"><p style="color: #000000 !important; font-size: 16px;">{st.session_state.summary}</p></div>', 
                unsafe_allow_html=True
            )
            
            # Statistics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Original Length", len(input_text.split()) if input_text else 0)
            with col_b:
                st.metric("Summary Length", len(st.session_state.summary.split()))
            with col_c:
                compression = round((1 - len(st.session_state.summary.split()) / len(input_text.split())) * 100, 1) if input_text else 0
                st.metric("Compression", f"{compression}%")
            
            # Audio player
            if st.session_state.audio_bytes:
                st.subheader("🔊 Audio Summary")
                st.audio(st.session_state.audio_bytes, format='audio/mp3')
                
                # Download button for audio
                st.download_button(
                    label="⬇️ Download Audio",
                    data=st.session_state.audio_bytes,
                    file_name="summary_audio.mp3",
                    mime="audio/mp3"
                )
            
            # Download button for summary text
            st.download_button(
                label="⬇️ Download Summary Text",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain"
            )
            
            # Clear button
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.summary = None
                st.session_state.audio_bytes = None
                st.rerun()
        else:
            st.info("👈 Enter text on the left side or upload a file, then press 'Generate Summary' button.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Made with ❤️ using Streamlit & Transformers</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
