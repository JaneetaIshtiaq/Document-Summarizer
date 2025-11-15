import streamlit as st
from transformers import pipeline
from gtts import gTTS
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
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

@st.cache_resource
def load_summarizer():
    """Load summarization pipeline"""
    try:
        # Use a model that works without issues
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # Use CPU
        )
        return summarizer
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

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
    with st.spinner("Loading AI model... This may take a few seconds."):
        summarizer = load_summarizer()
    
    if summarizer is None:
        st.error("❌ Model could not be loaded. Please check the console for errors.")
        return
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    max_length = st.sidebar.slider("Summary Maximum Length", 50, 300, 150, 10)
    min_length = st.sidebar.slider("Summary Minimum Length", 20, 100, 50, 10)
    
    # Ensure min_length is less than max_length
    if min_length >= max_length:
        st.sidebar.warning("⚠️ Minimum length must be less than maximum length!")
        min_length = max_length - 10
    
    audio_lang = st.sidebar.selectbox(
        "Audio Language",
        ["en", "ur", "hi"],
        format_func=lambda x: {"en": "English", "ur": "Urdu", "hi": "Hindi"}[x]
    )
    
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
                else:
                    st.error("❌ Could not extract text from the file.")
        
        # Generate summary button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if input_text and len(input_text.strip()) > 50:
                with st.spinner("Generating summary... This may take a few seconds."):
                    try:
                        # Handle long text by splitting
                        if len(input_text) > 1000:
                            chunks = [input_text[i:i+1000] for i in range(0, len(input_text), 1000)]
                            summaries = []
                            for chunk in chunks:
                                summary = summarizer(
                                    chunk,
                                    max_length=max_length,
                                    min_length=min_length,
                                    do_sample=False
                                )
                                summaries.append(summary[0]['summary_text'])
                            final_summary = ' '.join(summaries)
                        else:
                            summary = summarizer(
                                input_text,
                                max_length=max_length,
                                min_length=min_length,
                                do_sample=False
                            )
                            final_summary = summary[0]['summary_text']
                        
                        st.session_state.summary = final_summary
                        
                        # Generate audio
                        with st.spinner("Generating audio..."):
                            audio_bytes = text_to_speech(final_summary, audio_lang)
                            if audio_bytes:
                                st.session_state.audio_bytes = audio_bytes
                        
                        st.success("✅ Summary and audio generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
            else:
                st.warning("⚠️ Please enter at least 50 characters of text.")
    
    with col2:
        st.header("📊 Output")
        
        if st.session_state.summary:
            # Display summary
            st.subheader("📋 Summary")
            st.markdown(
                f'<div class="summary-box">{st.session_state.summary}</div>', 
                unsafe_allow_html=True
            )
            
            # Statistics
            if input_text:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Original Length", f"{len(input_text.split())} words")
                with col_b:
                    st.metric("Summary Length", f"{len(st.session_state.summary.split())} words")
                with col_c:
                    compression = round((1 - len(st.session_state.summary.split()) / len(input_text.split())) * 100, 1)
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
                    mime="audio/mp3",
                    use_container_width=True
                )
            
            # Download button for summary text
            st.download_button(
                label="⬇️ Download Summary Text",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
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
