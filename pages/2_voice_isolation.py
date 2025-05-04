import sys

import streamlit as st

sys.path.append(".")

import atexit
import os
import tempfile

import librosa
import numpy as np
import torch

from src.audio_processor import AudioProcessor
from src.utils.audio_viz import display_audio_visualizations
from src.utils.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

st.set_page_config(page_title="Speech Enhancement", page_icon="🎤", layout="wide")


# Initialize audio processor
@st.cache_resource
def get_audio_processor():
    return AudioProcessor()


def cleanup_files():
    """Clean up temporary files"""
    if "temp_files" in st.session_state:
        for filepath in st.session_state.temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temporary file: {filepath}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temporary file: {filepath}", exc_info=True
                )
        st.session_state.temp_files = []


# Register cleanup on exit
atexit.register(cleanup_files)

# Initialize session state
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []
if "enhanced_audio" not in st.session_state:
    st.session_state.enhanced_audio = None
if "process_audio" not in st.session_state:
    st.session_state.process_audio = False
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "enhanced_path" not in st.session_state:
    st.session_state.enhanced_path = None


def process_audio_file(filepath, chunk_size, overlap):
    """Process audio file and store enhanced audio in session state"""
    logger.info(
        f"Starting audio processing with chunk_size: {chunk_size}, overlap: {overlap}"
    )

    audio_processor = get_audio_processor()
    enhanced_path = audio_processor.process_audio(
        filepath, chunk_size=chunk_size, overlap=overlap
    )
    st.session_state.temp_files.append(enhanced_path)
    st.session_state.enhanced_path = enhanced_path

    # Store enhanced audio data in session state
    with open(enhanced_path, "rb") as file:
        st.session_state.enhanced_audio = file.read()

    logger.info("Audio processing completed successfully")
    return enhanced_path


def main():
    logger.info("Starting Speech Enhancement page")
    st.title("Speech Enhancement")
    st.markdown("""
    Enhance speech quality in interview recordings by reducing background noise.
    Uses Facebook's Denoiser model for fast, high-quality speech enhancement.
    Upload an audio file to get started.
    """)

    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")

        # Processing settings
        st.subheader("Processing")
        chunk_size = st.select_slider(
            "Chunk Size (samples)",
            options=[8000, 16000, 32000, 48000],
            value=16000,
            help="Size of audio chunks to process. Larger chunks may improve quality but use more memory. Values in samples at 16kHz (e.g., 16000 = 1 second).",
        )

        overlap = st.slider(
            "Chunk Overlap",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Overlap between chunks to prevent artifacts. Higher values may improve quality but increase processing time.",
        )

        st.info(
            "Using GPU for processing"
            if torch.cuda.is_available()
            else "Using CPU for processing"
        )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file", type=["mp3", "wav", "ogg", "flac"]
    )

    if uploaded_file is not None:
        logger.info(f"Processing uploaded file: {uploaded_file.name}")
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
            st.session_state.temp_files.append(tmp_filepath)
            logger.debug(f"Saved uploaded file to temporary path: {tmp_filepath}")

        # Only process if file changed or not processed yet
        if (
            st.session_state.processed_file != uploaded_file.name
            or st.session_state.enhanced_audio is None
        ):
            st.info("Processing audio... This may take a moment.")
            st.session_state.process_audio = True
            st.session_state.processed_file = uploaded_file.name

        try:
            # Process audio if needed
            if st.session_state.process_audio:
                with st.spinner("Enhancing speech..."):
                    process_audio_file(tmp_filepath, chunk_size, overlap)
                    st.success("Audio processing complete!")
                    st.session_state.process_audio = False

            # Only show visualizations if we have processed audio
            if st.session_state.enhanced_path is not None:
                # Load audio for visualization
                y_original, sr_original = librosa.load(tmp_filepath, sr=None)
                y_enhanced, sr_enhanced = librosa.load(
                    st.session_state.enhanced_path, sr=None
                )

                # Display waveforms and spectrograms in parallel
                st.subheader("Audio Comparison")
                col1, col2 = st.columns(2)

                with col1:
                    display_audio_visualizations(
                        y_original,
                        sr_original,
                        "Original Audio",
                        audio_data=uploaded_file.getvalue(),
                        audio_format=f"audio/{uploaded_file.name.split('.')[-1]}",
                    )

                with col2:
                    display_audio_visualizations(
                        y_enhanced,
                        sr_enhanced,
                        "Enhanced Speech",
                        audio_data=st.session_state.enhanced_audio,
                        audio_format="audio/wav",
                    )

                # Download button using session state data
                st.subheader("Download Processed Audio")
                st.download_button(
                    label="Download Enhanced Speech",
                    data=st.session_state.enhanced_audio,
                    file_name=f"enhanced_{uploaded_file.name.split('.')[0]}.wav",
                    mime="audio/wav",
                )

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            st.error(f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    main()
