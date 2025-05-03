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

# Get module logger
logger = get_logger(__name__)

st.set_page_config(
    page_title="Voice Isolation",
    page_icon="🎤",
    layout="wide"
)

# Initialize audio processor
@st.cache_resource
def get_audio_processor():
    return AudioProcessor()

def main():
    logger.info("Starting Voice Isolation page")
    st.title("Voice Isolation")
    st.markdown("""
    Clean audio files by isolating vocals and removing background noise.
    Upload an audio file to get started.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        model_type = st.selectbox(
            "Select Model",
            ["htdemucs", "htdemucs_ft", "mdx_extra"],
            index=0,
            help="htdemucs is faster, mdx_extra has better quality but is slower"
        )
        
        st.divider()
        st.markdown("### Advanced Settings")
        shifts = st.slider("Shifts (higher = better quality, slower)", 0, 10, 2,
                          help="Number of random shifts for augmenting separation")
        overlap = st.slider("Overlap ratio", 0.0, 0.5, 0.25, 
                           help="Overlap between the splits")
    
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
        
        with st.spinner("Separating vocals..."):
            try:
                logger.info(f"Starting audio processing with model: {model_type}, shifts: {shifts}, overlap: {overlap}")
                # Process the audio
                vocals_path = audio_processor.process_audio(
                    tmp_filepath, 
                    model_name=model_type,
                    shifts=shifts,
                    overlap=overlap
                )
                logger.info("Audio processing completed successfully")
                
                # Load the processed audio for visualization
                y_original, sr_original = librosa.load(tmp_filepath, sr=None)
                y_vocals, sr_vocals = librosa.load(vocals_path, sr=None)
                
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
                    st.markdown("**Isolated Vocals**")
                    fig, ax = plt.subplots(figsize=(10, 2))
                    librosa.display.waveshow(y_vocals, sr=sr_vocals, ax=ax)
                    ax.set_title("Isolated Vocals")
                    st.pyplot(fig)
                    
                    # Processed audio player
                    st.audio(vocals_path, format="audio/wav")
                
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
                    st.markdown("**Isolated Vocals Spectrogram**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_vocals)), ref=np.max)
                    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr_vocals, ax=ax)
                    ax.set_title("Isolated Vocals Spectrogram")
                    st.pyplot(fig)
                
                # Download button
                st.subheader("Download Processed Audio")
                with open(vocals_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Isolated Vocals",
                        data=file,
                        file_name=f"vocals_{uploaded_file.name.split('.')[0]}.wav",
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
