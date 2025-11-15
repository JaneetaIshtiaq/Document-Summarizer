import streamlit as st
import re
from gtts import gTTS
from io import BytesIO
import PyPDF2
import docx

st.set_page_config(page_title="Text Summarizer", page_icon="📄")
st.title("📄 Text Summarizer")

def simple_summary(text, sentences=3):
    """Simple extractive summarization"""
    sentences_list = re.split(r'[.!?]+', text)
    sentences_list = [s.strip() for s in sentences_list if len(s.strip()) > 10]
    
    if len(sentences_list) <= sentences:
        return text
    
    # Take first, middle and last sentences
    important_sentences = []
    important_sentences.append(sentences_list[0])  # First sentence
    if len(sentences_list) > 2:
        important_sentences.append(sentences_list[len(sentences_list)//2])  # Middle
    important_sentences.append(sentences_list[-1])  # Last sentence
    
    return '. '.join(important_sentences[:sentences]) + '.'

def main():
    text = st.text_area("Enter text:", height=150)
    
    if st.button("Generate Summary") and text:
        summary = simple_summary(text)
        st.success("Summary:")
        st.write(summary)
        
        # Audio
        try:
            tts = gTTS(text=summary, lang='en', slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            st.audio(audio_bytes, format='audio/mp3')
        except:
            st.info("Audio generation failed")

if __name__ == "__main__":
    main()
