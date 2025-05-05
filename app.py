# Add src directory to Python path
import sys

import streamlit as st

from src.utils.logging_config import get_logger, setup_logging

sys.path.append(".")

# Initialize logging
setup_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="Audio Processing App", page_icon="🎵", layout="wide")


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
    
    3. **Speaker Diarization**
       - Identify and segment different speakers
       - Generate RTTM files with speaker timings
       - Visualize speaker segments
    
    4. **Audio Transcription**
       - Transcribe Tamil-English code-mixed speech
       - Support for non-native English accents
       - Speaker-wise transcription with timestamps
    
    ### 📝 Getting Started
    
    Choose a tool to get started:
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            "🎵 Audio Chunking",
            use_container_width=True,
            help="Split long recordings into smaller segments",
        ):
            logger.info("Navigating to Audio Chunking page")
            st.switch_page("pages/1_audio_chunking.py")

    with col2:
        if st.button(
            "🎤 Voice Isolation",
            use_container_width=True,
            help="Extract clean vocals from mixed audio",
        ):
            logger.info("Navigating to Voice Isolation page")
            st.switch_page("pages/2_voice_isolation.py")
            
    with col3:
        if st.button(
            "👥 Speaker Diarization",
            use_container_width=True,
            help="Identify and segment different speakers",
        ):
            logger.info("Navigating to Speaker Diarization page")
            st.switch_page("pages/3_speaker_diarization.py")
            
    with col4:
        if st.button(
            "📝 Transcription",
            use_container_width=True,
            help="Transcribe Tamil-English code-mixed speech",
        ):
            logger.info("Navigating to Transcription page")
            st.switch_page("pages/4_transcription.py")

    st.markdown("""
    """)

    st.markdown("---")
    st.markdown("Built with Streamlit and Demucs")


if __name__ == "__main__":
    main()
