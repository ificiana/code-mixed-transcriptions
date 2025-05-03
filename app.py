import streamlit as st
from src.utils.logging_config import setup_logging, get_logger

# Add src directory to Python path
import sys
sys.path.append(".")

# Initialize logging
setup_logging()
logger = get_logger(__name__)

st.set_page_config(
    page_title="Audio Processing App",
    page_icon="🎵",
    layout="wide"
)

def main():
    logger.info("Starting Audio Processing App")
    st.title("Audio Processing App")
    
    st.markdown("""
    Welcome to the Audio Processing App! This application provides tools for working with audio files:
    
    ### 🎵 Features
    
    1. **Audio Chunking**
       - Split long audio files into smaller chunks
       - Customize chunk duration
       - Download individual chunks
    
    2. **Voice Isolation**
       - Isolate vocals from audio files
       - Remove background noise
       - Multiple model options for different quality/speed tradeoffs
    
    ### 📝 Getting Started
    
    Choose a tool to get started:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎵 Audio Chunking", use_container_width=True, help="Split long recordings into smaller segments"):
            logger.info("Navigating to Audio Chunking page")
            st.switch_page("pages/1_audio_chunking.py")
    
    with col2:
        if st.button("🎤 Voice Isolation", use_container_width=True, help="Extract clean vocals from mixed audio"):
            logger.info("Navigating to Voice Isolation page")
            st.switch_page("pages/2_voice_isolation.py")
            
    st.markdown("""
    """)
    
    st.markdown("---")
    st.markdown("Built with Streamlit and Demucs")

if __name__ == "__main__":
    main()
