import sys
sys.path.append(".")

import atexit
import os
import json
import tempfile
from pathlib import Path

import joblib
import librosa
import streamlit as st
import torch
from transformers import AutoModelForCTC, AutoProcessor

from src.utils.audio_utils import resample_audio
from src.utils.file_mapping import FileMapper
from src.utils.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

st.set_page_config(page_title="Transcription", page_icon="📝", layout="wide")

# Initialize cache directory
CACHE_DIR = Path("transcription_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize results directory
RESULTS_DIR = Path("transcription_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize file mapper
file_mapper = FileMapper(CACHE_DIR)

@st.cache_resource
def get_model():
    """Initialize and cache the MMS model"""
    try:
        # Use small model variant for CPU
        model_name = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCTC.from_pretrained(model_name)
        
        return model, processor
    except Exception as e:
        st.error(f"""
        **Error Loading Model**
        
        {str(e)}
        
        Make sure you have:
        1. Installed all requirements: pip install -r requirements.txt
        2. Have sufficient disk space for model download
        """)
        return None, None

def cleanup_files():
    """Clean up temporary files"""
    if "temp_files" in st.session_state:
        for filepath in st.session_state.temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Cleaned up temporary file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {filepath}", exc_info=True)
        st.session_state.temp_files = []

# Register cleanup on exit
atexit.register(cleanup_files)

# Initialize session state
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []
if "transcription_output" not in st.session_state:
    st.session_state.transcription_output = None
if "process_audio" not in st.session_state:
    st.session_state.process_audio = False
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

def get_cache_path(filepath, original_name):
    """Generate cache path for transcription results"""
    file_hash = joblib.hash(original_name + "_" + open(filepath, "rb").read().hex())
    return CACHE_DIR / f"{file_hash}.json"

def load_rttm(rttm_path):
    """Load speaker segments from RTTM file
    
    RTTM format:
    SPEAKER file_id channel start duration <NA> <NA> speaker_id <NA> <NA>
    Note: file_id may contain spaces, so we count positions from the end
    """
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                try:
                    # Count from end since last fields are fixed
                    # Format: ... 1 start duration <NA> <NA> speaker_id <NA> <NA>
                    start = float(parts[-7])      # 7th from end is start time
                    duration = float(parts[-6])   # 6th from end is duration
                    speaker = parts[-3]           # 3rd from end is speaker_id
                    
                    segments.append({
                        'start': start,
                        'duration': duration,
                        'speaker': speaker
                    })
                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping invalid RTTM line: {line.strip()}")
                    continue
    
    if not segments:
        raise ValueError("No valid speaker segments found in RTTM file")
    
    # Sort segments by start time
    segments.sort(key=lambda x: x['start'])
    
    return segments

def process_segment(audio_data, start_sample, end_sample, model, processor):
    """Process a single audio segment"""
    # Extract segment
    segment = audio_data[start_sample:end_sample]
    
    # Process through model
    inputs = processor(segment, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Decode predictions
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

def process_audio_file(audio_path, rttm_path, original_filename=None):
    """Process audio file with transcription
    
    Args:
        audio_path: Path to audio file
        rttm_path: Path to RTTM file with speaker segments
        original_filename: Original name of the uploaded file
    """
    logger.info(f"Starting transcription for file: {audio_path}")
    
    # Use original filename if provided
    original_name = original_filename if original_filename else os.path.basename(audio_path)
    
    # Check cache
    cache_file = file_mapper.get_mapping(original_name, "transcription")
    if cache_file and Path(cache_file).exists():
        logger.info("Loading cached transcription results")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Get model
    model, processor = get_model()
    if model is None or processor is None:
        raise ValueError("Could not initialize transcription model")
    
    # Resample audio to 16kHz
    resampled_path, _ = resample_audio(
        audio_path, 
        target_sr=16000, 
        cache_dir=CACHE_DIR,
        file_mapper=file_mapper,
        original_filename=original_name
    )
    
    # Load resampled audio
    audio_data, sr = librosa.load(resampled_path, sr=16000)
    
    # Load speaker segments
    segments = load_rttm(rttm_path)
    
    # Process each segment
    results = []
    total_segments = len(segments)
    
    for i, segment in enumerate(segments):
        # Update progress
        progress = (i + 1) / total_segments
        st.progress(progress)
        
        # Process segment
        start_sample = int(segment['start'] * sr)
        end_sample = int((segment['start'] + segment['duration']) * sr)
        
        transcription = process_segment(audio_data, start_sample, end_sample, model, processor)
        
        results.append({
            'start': segment['start'],
            'end': segment['start'] + segment['duration'],
            'speaker': segment['speaker'],
            'text': transcription
        })
    
    # Cache results
    cache_path = get_cache_path(audio_path, original_name)
    with open(cache_path, 'w') as f:
        json.dump(results, f, indent=2)
    file_mapper.add_mapping(original_name, str(cache_path), "transcription")
    
    return results

def main():
    logger.info("Starting Transcription page")
    st.title("Audio Transcription")
    st.markdown("""
    Transcribe audio with speaker diarization using Facebook's MMS model.
    Handles Tamil-English code-mixing and non-native English accents.
    Upload an audio file and its corresponding RTTM file to get started.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Cache Management")
        if st.button("Clear Transcription Cache"):
            # Clear transcription results cache
            for file in CACHE_DIR.glob("*.json"):
                try:
                    file.unlink()
                    logger.debug(f"Removed cached result: {file}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file: {file}", exc_info=True)
            # Clear file mappings for cache
            file_mapper.clear_mappings("transcription")
            st.success("Transcription cache cleared!")
    
    # File uploaders
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
    rttm_file = st.file_uploader("Upload RTTM file from diarization", type=["rttm"])
    
    if audio_file is not None and rttm_file is not None:
        logger.info(f"Processing files: {audio_file.name}, {rttm_file.name}")
        
        # Save uploaded files to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_audio:
            tmp_audio.write(audio_file.getvalue())
            audio_path = tmp_audio.name
            st.session_state.temp_files.append(audio_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".rttm") as tmp_rttm:
            tmp_rttm.write(rttm_file.getvalue())
            rttm_path = tmp_rttm.name
            st.session_state.temp_files.append(rttm_path)
        
        # Only process if files changed or not processed yet
        if (st.session_state.processed_file != audio_file.name or 
            st.session_state.transcription_output is None):
            st.info("Processing audio... This may take a while.")
            st.session_state.process_audio = True
            st.session_state.processed_file = audio_file.name
        
        try:
            # Process audio if needed
            if st.session_state.process_audio:
                st.markdown("### Processing Audio")
                results = process_audio_file(
                    audio_path,
                    rttm_path,
                    original_filename=audio_file.name
                )
                st.session_state.transcription_output = results
                st.success("Transcription complete!")
                st.session_state.process_audio = False
            
            # Display results
            if st.session_state.transcription_output is not None:
                st.subheader("Transcription Results")
                
                # Display results in table format
                results_df = []
                for segment in st.session_state.transcription_output:
                    results_df.append({
                        "Start Time": f"{segment['start']:.2f}s",
                        "End Time": f"{segment['end']:.2f}s",
                        "Speaker": segment['speaker'],
                        "Text": segment['text']
                    })
                st.table(results_df)
                
                # Download options
                st.subheader("Download Results")
                
                # JSON download
                json_str = json.dumps(st.session_state.transcription_output, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{os.path.splitext(audio_file.name)[0]}_transcription.json",
                    mime="application/json"
                )
                
                # CSV download
                csv_data = "Start Time,End Time,Speaker,Text\n"
                for segment in st.session_state.transcription_output:
                    csv_data += f"{segment['start']:.2f},{segment['end']:.2f},{segment['speaker']},{segment['text']}\n"
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{os.path.splitext(audio_file.name)[0]}_transcription.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            st.error(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()
