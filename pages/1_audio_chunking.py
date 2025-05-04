import streamlit as st
import sys
sys.path.append(".")

from src.utils.logging_config import get_logger
from src.utils.audio_viz import display_audio_visualizations
from src.audio_processor import AudioProcessor
import librosa
import os
import tempfile

# Get module logger
logger = get_logger(__name__)

st.set_page_config(
    page_title="Audio Chunking",
    page_icon="✂️",
    layout="wide"
)

def cleanup_old_chunks():
    """Clean up old chunk files"""
    if os.path.exists("temp_chunks"):
        logger.info("Cleaning up old chunk files")
        for file in os.listdir("temp_chunks"):
            try:
                file_path = os.path.join("temp_chunks", file)
                os.remove(file_path)
                logger.debug(f"Removed old chunk file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove chunk file {file}: {str(e)}", exc_info=True)

# Initialize audio processor with a fixed output directory
@st.cache_resource
def get_audio_processor():
    output_dir = "temp_chunks"
    os.makedirs(output_dir, exist_ok=True)
    return AudioProcessor(output_dir=output_dir)

# Initialize session state
if 'chunk_paths' not in st.session_state:
    st.session_state.chunk_paths = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'chunk_data' not in st.session_state:
    st.session_state.chunk_data = {}
if 'create_chunks' not in st.session_state:
    st.session_state.create_chunks = False

def process_chunks(filepath, duration):
    """Process audio into chunks and store in session state"""
    logger.info(f"Starting audio chunking with duration: {duration} seconds")
    
    # Clean up old chunks before processing new ones
    cleanup_old_chunks()
    
    audio_processor = get_audio_processor()
    st.session_state.chunk_paths = audio_processor.chunk_audio(filepath, duration)
    
    # Store chunk data in session state
    st.session_state.chunk_data = {}
    for i, chunk_path in enumerate(st.session_state.chunk_paths, 1):
        with open(chunk_path, "rb") as file:
            st.session_state.chunk_data[i] = file.read()
    
    num_chunks = len(st.session_state.chunk_paths)
    logger.info(f"Successfully created {num_chunks} chunks")
    return num_chunks

def main():
    logger.info("Starting Audio Chunking page")
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
        logger.info(f"Processing uploaded file: {uploaded_file.name}")
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
            logger.debug(f"Saved uploaded file to temporary path: {tmp_filepath}")
        
        try:
            # Load and display original audio
            y_original, sr_original = librosa.load(tmp_filepath, sr=None)
            duration = len(y_original) / sr_original
            
            st.subheader("Original Audio")
            display_audio_visualizations(
                y_original,
                sr_original,
                "Original Audio",
                audio_data=uploaded_file.getvalue(),
                audio_format=f"audio/{uploaded_file.name.split('.')[-1]}"
            )
            st.markdown(f"**Duration:** {duration:.2f} seconds")
            st.markdown(f"**Sample Rate:** {sr_original} Hz")
            
            # Only process if file changed or no chunks exist
            if st.session_state.processed_file != uploaded_file.name or st.session_state.chunk_paths is None:
                if st.button("Create Chunks"):
                    st.session_state.create_chunks = True
                    st.session_state.processed_file = uploaded_file.name
            
            # Process chunks if button was clicked
            if st.session_state.create_chunks:
                st.info("Creating audio chunks...")
                num_chunks = process_chunks(tmp_filepath, chunk_duration)
                st.success(f"Created {num_chunks} chunks!")
                st.session_state.create_chunks = False
            
            # Display chunks if they exist
            if st.session_state.chunk_paths:
                st.subheader("Audio Chunks")
                for i, chunk_path in enumerate(st.session_state.chunk_paths, 1):
                    with st.expander(f"Chunk {i}"):
                        # Load chunk for visualization
                        y_chunk, sr_chunk = librosa.load(chunk_path, sr=None)
                        chunk_duration = len(y_chunk) / sr_chunk
                        
                        # Display chunk visualizations
                        display_audio_visualizations(
                            y_chunk,
                            sr_chunk,
                            f"Chunk {i}",
                            audio_data=st.session_state.chunk_data[i],
                            audio_format="audio/wav"
                        )
                        st.markdown(f"**Duration:** {chunk_duration:.2f} seconds")
                        
                        # Download button using session state data
                        st.download_button(
                            label=f"Download Chunk {i}",
                            data=st.session_state.chunk_data[i],
                            file_name=f"chunk_{i}_{uploaded_file.name.split('.')[0]}.wav",
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
