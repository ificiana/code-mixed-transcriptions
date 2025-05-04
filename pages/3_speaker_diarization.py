import sys

sys.path.append(".")

import atexit
import os
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import streamlit as st
import torch

from src.utils.audio_utils import resample_audio
from src.utils.audio_viz import display_audio_visualizations
from src.utils.file_mapping import FileMapper
from src.utils.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

st.set_page_config(page_title="Speaker Diarization", page_icon="👥", layout="wide")

# Initialize cache directory
CACHE_DIR = Path("diarization_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize results directory
RESULTS_DIR = Path("diarization_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize file mapper
file_mapper = FileMapper(CACHE_DIR)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the diarization pipeline"""
    try:
        from pyannote.audio import Pipeline

        # Load pipeline from cache
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        return pipeline
    except Exception as e:
        st.error(f"""
        **Error Loading Pipeline**
        
        {str(e)}
        
        Make sure you have:
        1. Installed huggingface-cli: pip install --upgrade huggingface-hub
        2. Logged in: huggingface-cli login
        3. Accepted the user conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
        4. Run the download script once:
           ```bash
           python scripts/download_models.py
           ```
        """)
        return None


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
if "diarization_output" not in st.session_state:
    st.session_state.diarization_output = None
if "process_audio" not in st.session_state:
    st.session_state.process_audio = False
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


def get_cache_path(filepath, original_name):
    """Generate cache path for diarization results
    
    Args:
        filepath: Path to the audio file
        original_name: Original filename for consistent mapping
    """
    # Use original name in hash to ensure consistent mapping
    file_hash = joblib.hash(original_name + "_" + open(filepath, "rb").read().hex())
    return CACHE_DIR / f"{file_hash}.pkl"


def save_rttm(diarization, output_path):
    """Save diarization output in RTTM format"""
    with open(output_path, "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            line = f"SPEAKER {os.path.basename(output_path)} 1 {turn.start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            f.write(line)


def update_progress(current, total):
    """Update progress bar and status"""
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
        st.session_state.status = st.empty()

    progress = float(current) / float(total)
    st.session_state.progress_bar.progress(progress)
    st.session_state.status.markdown(f"""
    **Processing Progress:**
    - {int(progress * 100)}% complete
    """)

    if current == total:
        st.session_state.status.empty()
        del st.session_state.progress_bar
        del st.session_state.status


def process_audio_file(filepath, num_speakers=None, original_filename=None):
    """Process audio file with speaker diarization

    Args:
        filepath: Path to audio file
        num_speakers: Expected number of speakers (None for auto-detection)
        original_filename: Original name of the uploaded file

    Notes:
        Audio is automatically resampled to 8kHz if sample rate is higher,
        which significantly speeds up processing without affecting accuracy.
    """
    logger.info(f"Starting diarization for file: {filepath}")

    # Use original filename if provided, otherwise fallback to basename
    original_name = original_filename if original_filename else os.path.basename(filepath)
    
    # Check cache using file mapper
    cache_file = file_mapper.get_mapping(original_name, "cache")
    if cache_file and Path(cache_file).exists():
        logger.info("Loading cached diarization results")
        return joblib.load(cache_file)

    # Get pipeline
    pipeline = get_pipeline()
    if pipeline is None:
        raise ValueError(
            "Could not initialize diarization pipeline. Please check the error message above "
            "and make sure pyannote.audio is installed correctly."
        )

    # Run diarization with user-selected parameters
    with st.spinner("Running speaker diarization... This may take a few minutes."):
        # Resample if needed
        resampled_path, _ = resample_audio(
            filepath, target_sr=16000, cache_dir=CACHE_DIR, file_mapper=file_mapper,
            original_filename=original_name
        )

        # Convert num_speakers=0 to None for auto-detection
        effective_num_speakers = None if num_speakers == 0 else num_speakers

        # Run diarization
        diarization = pipeline(resampled_path, num_speakers=effective_num_speakers)

    # Cache results with mapping using original name for consistency
    cache_path = get_cache_path(resampled_path, original_name)
    joblib.dump(diarization, cache_path)
    file_mapper.add_mapping(original_name, str(cache_path), "cache")

    # Save RTTM with mapping
    rttm_path = RESULTS_DIR / f"{os.path.splitext(original_name)[0]}.rttm"
    save_rttm(diarization, rttm_path)
    file_mapper.add_mapping(original_name, str(rttm_path), "rttm")

    return diarization


def plot_diarization(fig, y, sr, diarization):
    """Add speaker segments visualization to audio plot"""
    import matplotlib.pyplot as plt

    # Get axis for waveform
    ax = fig.axes[0]

    # Plot colored regions for each speaker
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(diarization.labels()))))
    color_map = dict(zip(diarization.labels(), colors))

    ylim = ax.get_ylim()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        color = color_map[speaker]
        ax.axvspan(start_sample, end_sample, alpha=0.2, color=color, label=speaker)

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.2)
        for color in color_map.values()
    ]
    ax.legend(handles, color_map.keys(), title="Speakers", loc="upper right")


def main():
    logger.info("Starting Speaker Diarization page")
    st.title("Speaker Diarization")
    st.markdown("""
    Identify and segment different speakers in audio recordings.
    Uses pyannote.audio's pretrained model for fast, accurate speaker diarization.
    Upload an audio file to get started.
    """)

    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")

        st.header("Processing Settings")
        num_speakers = st.slider(
            "Number of Speakers",
            min_value=0,
            max_value=10,
            value=None,
            help="Expected number of speakers. Set to 0 for auto-detect.",
        )

        st.header("Display Settings")
        enable_viz = st.checkbox(
            "Enable Visualizations",
            value=True,
            help="Show waveform with speaker segments. May be slow for long files.",
        )

        st.info(
            "Using GPU for processing (faster)"
            if torch.cuda.is_available()
            else "Using CPU for processing (slower)"
        )

        # Cache clearing buttons
        st.header("Cache Management")
        if st.button("Clear Results Cache"):
            # Clear diarization results cache
            for file in CACHE_DIR.glob("*.pkl"):
                try:
                    file.unlink()
                    logger.debug(f"Removed cached result: {file}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file: {file}", exc_info=True)
            # Clear file mappings for cache
            file_mapper.clear_mappings("cache")
            st.success("Results cache cleared!")

        if st.button("Clear Resampled Audio Cache"):
            # Clear resampled audio files
            for file in CACHE_DIR.glob("*.wav"):
                try:
                    file.unlink()
                    logger.debug(f"Removed resampled audio: {file}")
                except Exception as e:
                    logger.error(f"Failed to remove resampled file: {file}", exc_info=True)
            # Clear file mappings for resampled
            file_mapper.clear_mappings("resampled")
            st.success("Resampled audio cache cleared!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file", type=["mp3", "wav", "ogg", "flac"]
    )

    if uploaded_file is not None:
        logger.info(f"Processing uploaded file: {uploaded_file.name}")

        # Save uploaded file to temporary file
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
            or st.session_state.diarization_output is None
        ):
            st.info("Processing audio... This may take a moment.")
            st.session_state.process_audio = True
            st.session_state.processed_file = uploaded_file.name

        try:
            # Process audio if needed
            if st.session_state.process_audio:
                st.markdown("### Processing Audio")
                diarization = process_audio_file(
                    tmp_filepath,
                    original_filename=uploaded_file.name,
                    num_speakers=num_speakers
                )
                st.session_state.diarization_output = diarization
                st.success("Audio processing complete!")
                st.session_state.process_audio = False

            # Display results if we have processed audio
            if st.session_state.diarization_output is not None:
                # Cache management for current file
                st.subheader("File Cache Management")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear This File's Results Cache"):
                        cache_file = file_mapper.get_mapping(uploaded_file.name, "cache")
                        if cache_file and Path(cache_file).exists():
                            Path(cache_file).unlink()
                            logger.debug(f"Removed cached result for: {uploaded_file.name}")
                            file_mapper.remove_mapping(uploaded_file.name, "cache")
                            st.session_state.diarization_output = None
                            st.success("File's results cache cleared!")
                            st.experimental_rerun()

                with col2:
                    if st.button("Clear This File's Audio Cache"):
                        resampled_file = file_mapper.get_mapping(uploaded_file.name, "resampled")
                        if resampled_file and Path(resampled_file).exists():
                            Path(resampled_file).unlink()
                            logger.debug(f"Removed resampled audio for: {uploaded_file.name}")
                            file_mapper.remove_mapping(uploaded_file.name, "resampled")
                            st.success("File's audio cache cleared!")
                            st.experimental_rerun()

                # Load audio for visualization
                y, sr = librosa.load(tmp_filepath, sr=None)

                # Display waveform with speaker segments
                st.subheader("Audio with Speaker Segments")
                fig = display_audio_visualizations(
                    y,
                    sr,
                    "Audio Waveform",
                    audio_data=uploaded_file.getvalue(),
                    audio_format=f"audio/{uploaded_file.name.split('.')[-1]}",
                    enable_viz=enable_viz,
                    return_fig=True,
                )

                if enable_viz and fig is not None:
                    plot_diarization(fig, y, sr, st.session_state.diarization_output)
                    st.pyplot(fig)

                # Display diarization timeline
                st.subheader("Speaker Timeline")
                timeline_data = []
                for turn, _, speaker in st.session_state.diarization_output.itertracks(
                    yield_label=True
                ):
                    timeline_data.append(
                        {
                            "speaker": speaker,
                            "start": f"{turn.start:.2f}s",
                            "end": f"{turn.end:.2f}s",
                            "duration": f"{turn.duration:.2f}s",
                        }
                    )
                st.json(timeline_data)

                # Download RTTM file
                st.subheader("Download Results")
                rttm_path = (
                    RESULTS_DIR / f"{os.path.splitext(uploaded_file.name)[0]}.rttm"
                )
                if rttm_path.exists():
                    with open(rttm_path, "r") as f:
                        rttm_content = f.read()
                    st.download_button(
                        label="Download RTTM File",
                        data=rttm_content,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}.rttm",
                        mime="text/plain",
                    )
                else:
                    st.warning(
                        "RTTM file not found. Please try processing the audio again."
                    )

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            st.error(f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    main()
