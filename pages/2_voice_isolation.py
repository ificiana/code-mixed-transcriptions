import streamlit as st
import sys
sys.path.append(".")

from src.utils.logging_config import get_logger
from src.audio_processor import AudioProcessor
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import tempfile
import torch

# Get module logger
logger = get_logger(__name__)

st.set_page_config(
    page_title="Speech Enhancement",
    page_icon="🎤",
    layout="wide"
)

# Initialize audio processor
@st.cache_resource
def get_audio_processor():
    return AudioProcessor()

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
            help="Size of audio chunks to process. Larger chunks may improve quality but use more memory. Values in samples at 16kHz (e.g., 16000 = 1 second)."
        )
        
        overlap = st.slider(
            "Chunk Overlap",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Overlap between chunks to prevent artifacts. Higher values may improve quality but increase processing time."
        )
        
        st.info("Using GPU for processing" if torch.cuda.is_available() else "Using CPU for processing")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac"])
    
    if uploaded_file is not None:
        logger.info(f"Processing uploaded file: {uploaded_file.name}")
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
            logger.debug(f"Saved uploaded file to temporary path: {tmp_filepath}")
        
        # Process the audio file
        st.info("Processing audio... This may take a moment.")
        
        audio_processor = get_audio_processor()
        
        with st.spinner("Enhancing speech..."):
            try:
                logger.info(f"Starting audio processing with chunk_size: {chunk_size}, overlap: {overlap}")
                # Process the audio
                enhanced_path = audio_processor.process_audio(
                    tmp_filepath,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                logger.info("Audio processing completed successfully")
                
                # Load the processed audio for visualization
                y_original, sr_original = librosa.load(tmp_filepath, sr=None)
                y_enhanced, sr_enhanced = librosa.load(enhanced_path, sr=None)
                
                st.success("Audio processing complete!")
                
                # Display waveforms
                st.subheader("Audio Waveforms")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Audio**")
                    fig, ax = plt.subplots(figsize=(10, 2))
                    librosa.display.waveshow(y_original, sr=sr_original, ax=ax)
                    ax.set_title("Original Audio")
                    st.pyplot(fig)
                    
                    # Original audio player
                    st.audio(tmp_filepath, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                
                with col2:
                    st.markdown("**Enhanced Speech**")
                    fig, ax = plt.subplots(figsize=(10, 2))
                    librosa.display.waveshow(y_enhanced, sr=sr_enhanced, ax=ax)
                    ax.set_title("Enhanced Speech")
                    st.pyplot(fig)
                    
                    # Processed audio player
                    st.audio(enhanced_path, format="audio/wav")
                
                # Spectrograms
                st.subheader("Spectrograms")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Audio Spectrogram**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max)
                    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr_original, ax=ax)
                    ax.set_title("Original Audio Spectrogram")
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**Enhanced Speech Spectrogram**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_enhanced)), ref=np.max)
                    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr_enhanced, ax=ax)
                    ax.set_title("Enhanced Speech Spectrogram")
                    st.pyplot(fig)
                
                # Download button
                st.subheader("Download Processed Audio")
                with open(enhanced_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Enhanced Speech",
                        data=file,
                        file_name=f"enhanced_{uploaded_file.name.split('.')[0]}.wav",
                        mime="audio/wav"
                    )
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}", exc_info=True)
                st.error(f"Error processing audio: {str(e)}")
            
            finally:
                # Clean up temporary files
                try:
                    os.remove(tmp_filepath)
                    logger.debug(f"Cleaned up temporary file: {tmp_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {tmp_filepath}", exc_info=True)

if __name__ == "__main__":
    main()
