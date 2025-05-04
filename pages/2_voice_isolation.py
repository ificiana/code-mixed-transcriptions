import sys

import streamlit as st

sys.path.append(".")

import atexit
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import torch
import torchaudio

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
if "start_time" not in st.session_state:
    st.session_state.start_time = None


def update_progress(current, total):
    """Update progress bar and status"""
    import time

    # Initialize progress tracking
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
        st.session_state.status = st.empty()
        st.session_state.start_time = time.time()

    progress = float(current) / float(total)
    percentage = int(progress * 100)

    # Calculate time metrics
    elapsed = time.time() - st.session_state.start_time
    if current > 1:  # Need at least 2 chunks to estimate
        chunks_per_second = (current - 1) / elapsed
        remaining_chunks = total - current
        eta = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
    else:
        eta = 0

    # Update progress bar
    st.session_state.progress_bar.progress(progress)

    # Update status message
    st.session_state.status.markdown(f"""
    **Processing Progress:**
    - Processing chunk {current} of {total}
    - {percentage}% complete
    - Time elapsed: {int(elapsed)}s
    - Estimated remaining: {int(eta)}s
    """)

    if current == total:
        final_time = time.time() - st.session_state.start_time
        st.session_state.status.markdown(f"""
        **Processing Complete!**
        - Total time: {int(final_time)}s
        - Average speed: {total/final_time:.1f} chunks/second
        """)
        time.sleep(2)  # Show final stats briefly
        st.session_state.status.empty()
        del st.session_state.progress_bar
        del st.session_state.status
        del st.session_state.start_time


def process_audio_file(filepath, chunk_size, overlap, max_workers):
    """Process audio file and store enhanced audio in session state"""
    logger.info(
        f"Starting audio processing with chunk_size: {chunk_size}, overlap: {overlap}, "
        f"max_workers: {max_workers}"
    )

    audio_processor = get_audio_processor()
    # Convert chunk size to duration (at 16kHz)
    chunk_duration = chunk_size / 16000  # seconds at 16kHz

    # Create chunks
    chunk_paths = audio_processor.chunk_audio(filepath, chunk_duration=chunk_duration)
    total_chunks = len(chunk_paths)
    logger.info(f"Split audio into {total_chunks} chunks")

    # Process chunks in parallel
    processed_chunks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(
                audio_processor._process_chunk,
                chunk_path,
                update_progress,
                i,
                total_chunks,
            ): i
            for i, chunk_path in enumerate(chunk_paths)
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                processed_chunks.append((chunk_idx, future.result()))
            except Exception as e:
                logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
                raise

    # Sort chunks by index
    processed_chunks.sort(key=lambda x: x[0])
    processed_audio = torch.cat([chunk for _, chunk in processed_chunks], dim=1)

    # Create output filename
    input_filename = os.path.basename(filepath)
    output_filename = f"enhanced_{os.path.splitext(input_filename)[0]}.wav"
    output_path = str(audio_processor.output_dir / output_filename)

    # Save processed audio
    torchaudio.save(output_path, processed_audio, 16000)  # Always save at 16kHz
    logger.info(f"Successfully saved enhanced audio to: {output_path}")
    enhanced_path = output_path
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

        st.header("Display Settings")
        enable_viz = st.checkbox(
            "Enable Visualizations",
            value=True,
            help="Show waveforms and spectrograms. May be slow for long files.",
        )

        chunk_size = st.select_slider(
            "Chunk Size (samples)",
            options=[8000, 16000, 32000, 48000],
            value=16000,
            help="Size of audio chunks to process. Larger chunks may improve quality but use more memory. Values in samples at 16kHz.",
        )

        overlap = st.slider(
            "Chunk Overlap",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Overlap between chunks to prevent artifacts. Higher values may improve quality but increase processing time.",
        )

        max_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=8,
            value=4,
            help="Number of parallel processing threads. Higher values may improve speed but use more memory.",
        )

        st.markdown("---")
        st.markdown(
            """
        **Processing Configuration:**
        - Chunk Size: {} samples ({:.1f}s at 16kHz)
        - Overlap: {}%
        - Parallel Workers: {}
        """.format(chunk_size, chunk_size / 16000, int(overlap * 100), max_workers)
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
                st.markdown("### Processing Audio")
                process_audio_file(tmp_filepath, chunk_size, overlap, max_workers)
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
                        enable_viz=enable_viz,
                    )

                with col2:
                    display_audio_visualizations(
                        y_enhanced,
                        sr_enhanced,
                        "Enhanced Speech",
                        audio_data=st.session_state.enhanced_audio,
                        audio_format="audio/wav",
                        enable_viz=enable_viz,
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
