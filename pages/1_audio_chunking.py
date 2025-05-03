import streamlit as st

st.set_page_config(
    page_title="Audio Chunking",
    page_icon="✂️",
    layout="wide"
)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import sys
sys.path.append(".")
from src.audio_processor import AudioProcessor

# Initialize audio processor
@st.cache_resource
def get_audio_processor():
    return AudioProcessor()

def main():
    st.title("Audio Chunking")
    st.markdown("""
    Split your audio files into smaller chunks. This is useful for processing long audio files
    or creating segments for easier management.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Chunking Settings")
        chunk_duration = st.slider(
            "Chunk Duration (seconds)", 
            10, 300, 30,
            help="Duration of each audio chunk"
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        try:
            # Load and display original audio
            y_original, sr_original = librosa.load(tmp_filepath, sr=None)
            duration = len(y_original) / sr_original
            
            st.subheader("Original Audio")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display waveform
                fig, ax = plt.subplots(figsize=(10, 2))
                librosa.display.waveshow(y_original, sr=sr_original, ax=ax)
                ax.set_title("Original Audio Waveform")
                st.pyplot(fig)
            
            with col2:
                st.markdown(f"**Duration:** {duration:.2f} seconds")
                st.markdown(f"**Sample Rate:** {sr_original} Hz")
                st.audio(tmp_filepath, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            
            # Create chunks
            if st.button("Create Chunks"):
                st.info("Creating audio chunks...")
                
                audio_processor = get_audio_processor()
                chunk_paths = audio_processor.chunk_audio(tmp_filepath, chunk_duration)
                
                st.success(f"Created {len(chunk_paths)} chunks!")
                
                # Display chunks
                st.subheader("Audio Chunks")
                for i, chunk_path in enumerate(chunk_paths, 1):
                    with st.expander(f"Chunk {i}"):
                        # Load chunk for visualization
                        y_chunk, sr_chunk = librosa.load(chunk_path, sr=None)
                        chunk_duration = len(y_chunk) / sr_chunk
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Display waveform
                            fig, ax = plt.subplots(figsize=(10, 2))
                            librosa.display.waveshow(y_chunk, sr=sr_chunk, ax=ax)
                            ax.set_title(f"Chunk {i} Waveform")
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown(f"**Duration:** {chunk_duration:.2f} seconds")
                            # Audio player
                            st.audio(chunk_path, format="audio/wav")
                            # Download button
                            with open(chunk_path, "rb") as file:
                                st.download_button(
                                    label=f"Download Chunk {i}",
                                    data=file,
                                    file_name=f"chunk_{i}_{uploaded_file.name.split('.')[0]}.wav",
                                    mime="audio/wav"
                                )
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        
        finally:
            # Clean up temporary files
            try:
                os.remove(tmp_filepath)
            except:
                pass

if __name__ == "__main__":
    main()
