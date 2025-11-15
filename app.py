import streamlit as st
from transformers import pipeline
from gtts import gTTS
import base64
from io import BytesIO
import PyPDF2
import docx

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
        border-left: 5px solid #1f77b4;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
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
def load_summarizer():
    """Load summarization pipeline"""
    try:
        # Using pipeline instead of direct model loading for better compatibility
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1  # Use CPU
        )
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
        return ""

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
        return ""

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
    with st.spinner("🔄 Loading AI model... This may take a few seconds."):
        summarizer = load_summarizer()
    
    if summarizer is None:
        st.error("❌ Model could not be loaded. Please check the requirements.")
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        max_length = st.slider(
            "Summary Maximum Length", 
            50, 300, 
            st.session_state.max_length, 
            10
        )
        
        min_length = st.slider(
            "Summary Minimum Length", 
            20, 100, 
            st.session_state.min_length, 
            10
        )
        
        # Ensure min_length is less than max_length
        if min_length >= max_length:
            st.warning("⚠️ Minimum length must be less than maximum length!")
            min_length = max_length - 10
        
        audio_lang = st.selectbox(
            "Audio Language",
            ["en", "ur", "hi"],
            format_func=lambda x: {"en": "English", "ur": "Urdu", "hi": "Hindi"}[x]
        )
        
        st.info(f"**Current Settings:**\n- Max Length: {max_length}\n- Min Length: {min_length}")

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Text Input", "File Upload"],
            horizontal=True
        )
        
        input_text = ""
        
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter your text here:",
                height=250,
                placeholder="Paste your document text here...\n\nExample: The quick brown fox jumps over the lazy dog. This is a sample text for summarization. Artificial intelligence is transforming how we process information and create summaries from large documents."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload document (PDF, DOCX, TXT)",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, Word documents, Text files"
            )
            
            if uploaded_file is not None:
                # Show file info
                st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size} bytes")
                
                with st.spinner("📖 Reading file content..."):
                    if uploaded_file.type == "application/pdf":
                        input_text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        input_text = extract_text_from_docx(uploaded_file)
                    else:
                        input_text = uploaded_file.read().decode('utf-8')
                
                if input_text:
                    st.success(f"✅ File successfully read! ({len(input_text)} characters)")
                    with st.expander("📋 Extracted Text Preview"):
                        st.text(input_text[:500] + "..." if len(input_text) > 500 else input_text)
                else:
                    st.error("❌ Could not extract text from the file.")
        
        # Generate summary button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if input_text and len(input_text.strip()) > 50:
                with st.spinner("🤖 Generating summary... This may take a few seconds."):
                    try:
                        # Handle long text by truncating
                        if len(input_text) > 1024:
                            st.info("📝 Long text detected. Using first 1024 characters for faster processing.")
                            processed_text = input_text[:1024]
                        else:
                            processed_text = input_text
                        
                        # Generate summary
                        summary_result = summarizer(
                            processed_text,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        
                        summary = summary_result[0]['summary_text']
                        st.session_state.summary = summary
                        
                        # Generate audio
                        with st.spinner("🔊 Generating audio..."):
                            audio_bytes = text_to_speech(summary, audio_lang)
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
                    original_words = len(input_text.split())
                    st.metric("Original Text", f"{original_words} words")
                with col_b:
                    summary_words = len(st.session_state.summary.split())
                    st.metric("Summary", f"{summary_words} words")
                with col_c:
                    compression = ((original_words - summary_words) / original_words) * 100
                    st.metric("Reduction", f"{compression:.1f}%")
            
            # Audio player
            if st.session_state.audio_bytes:
                st.subheader("🔊 Audio Summary")
                st.audio(st.session_state.audio_bytes, format='audio/mp3')
                
                # Download buttons
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="⬇️ Download Audio",
                        data=st.session_state.audio_bytes,
                        file_name="summary_audio.mp3",
                        mime="audio/mp3",
                        use_container_width=True
                    )
                with col_dl2:
                    st.download_button(
                        label="⬇️ Download Text",
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
            st.info(
                "👆 **How to use:**\n"
                "1. Enter text or upload a file on the left\n"
                "2. Adjust settings in the sidebar\n"
                "3. Click 'Generate Summary' button\n"
                "4. Get your summary with audio!"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Made with ❤️ using Streamlit, Transformers & gTTS"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
