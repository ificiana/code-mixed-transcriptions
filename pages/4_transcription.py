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
def get_models():
    """Initialize and cache MMS model for transcription"""
    try:
        # Load the base MMS model
        model_name = "facebook/mms-1b-all"
        
        # Initialize a single model and processor
        model = AutoModelForCTC.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        return model, model, processor
    except Exception as e:
        st.error(f"""
        **Error Loading Models**
        
        {str(e)}
        
        Make sure you have:
        1. Installed all requirements: pip install -r requirements.txt
        2. Have sufficient disk space for model downloads
        """)
        return None, None, None

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

def process_segment(audio_data, start_sample, end_sample, models, processor, speaker_history):
    """Process a single audio segment with language detection"""
    # Extract segment
    segment = audio_data[start_sample:end_sample]
    
    # Process audio with the model
    inputs = processor(segment, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Process with the model
        outputs = models[0](**inputs)
    
    # Calculate confidence scores
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    conf = probs.max().item()
    
    # Get speaker's language history if available
    speaker = speaker_history.get('current_speaker')
    speaker_data = speaker_history.get(speaker, {'lang_counts': {}, 'dominant_lang': None})
    dominant_lang = speaker_data.get('dominant_lang')
    
    # For simplicity, we'll use speaker history to determine language
    # If no history, default to English
    lang = dominant_lang if dominant_lang else 'eng'
    
    # Set confidence values for reporting
    tamil_conf = conf if lang == 'tam' else 0.0
    english_conf = conf if lang == 'eng' else 0.0
    
    # Update speaker history
    if speaker:
        if speaker not in speaker_history:
            speaker_history[speaker] = {'lang_counts': {}, 'dominant_lang': None}
        
        speaker_history[speaker]['lang_counts'][lang] = speaker_history[speaker]['lang_counts'].get(lang, 0) + 1
        
        # Update dominant language if needed
        lang_counts = speaker_history[speaker]['lang_counts']
        max_lang = max(lang_counts.items(), key=lambda x: x[1])[0]
        speaker_history[speaker]['dominant_lang'] = max_lang
    
    # Get transcription
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return {
        'text': transcription,
        'language': lang,
        'tamil_confidence': tamil_conf,
        'english_confidence': english_conf
    }

def process_audio_file(audio_path, rttm_path, original_filename=None):
    """Process audio file with transcription using hybrid language detection"""
    logger.info(f"Starting transcription for file: {audio_path}")
    
    # Use original filename if provided
    original_name = original_filename if original_filename else os.path.basename(audio_path)
    
    # Check cache
    cache_file = file_mapper.get_mapping(original_name, "transcription")
    if cache_file and Path(cache_file).exists():
        logger.info("Loading cached transcription results")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Get models
    tamil_model, english_model, processor = get_models()
    if None in (tamil_model, english_model, processor):
        raise ValueError("Could not initialize transcription models")
    
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
    
    # Initialize speaker history
    speaker_history = {}
    
    # Process each segment
    results = []
    total_segments = len(segments)
    
    for i, segment in enumerate(segments):
        # Set current speaker for history tracking
        speaker_history['current_speaker'] = segment['speaker']
        # Update progress
        progress = (i + 1) / total_segments
        st.progress(progress)
        
        # Process segment
        start_sample = int(segment['start'] * sr)
        end_sample = int((segment['start'] + segment['duration']) * sr)
        
        segment_result = process_segment(
            audio_data, 
            start_sample, 
            end_sample, 
            (tamil_model, english_model),
            processor,
            speaker_history
        )
        
        results.append({
            'start': segment['start'],
            'end': segment['start'] + segment['duration'],
            'speaker': segment['speaker'],
            'text': segment_result['text'],
            'language': segment_result['language'],
            'confidence': {
                'tamil': segment_result['tamil_confidence'],
                'english': segment_result['english_confidence']
            }
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
                
                # Display results in table format with language info
                results_df = []
                for segment in st.session_state.transcription_output:
                    results_df.append({
                        "Start Time": f"{segment['start']:.2f}s",
                        "End Time": f"{segment['end']:.2f}s",
                        "Speaker": segment['speaker'],
                        "Language": "Tamil" if segment['language'] == 'tam' else "English",
                        "Text": segment['text'],
                        "Confidence": f"Ta: {segment['confidence']['tamil']:.2f}, En: {segment['confidence']['english']:.2f}"
                    })
                st.table(results_df)
                
                # Display speaker language statistics
                st.subheader("Speaker Language Statistics")
                speaker_stats = {}
                for segment in st.session_state.transcription_output:
                    speaker = segment['speaker']
                    lang = segment['language']
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {'tam': 0, 'eng': 0}
                    speaker_stats[speaker][lang] += 1
                
                for speaker, stats in speaker_stats.items():
                    total = stats['tam'] + stats['eng']
                    st.write(f"**Speaker {speaker}:**")
                    st.write(f"- Tamil: {stats['tam']} segments ({stats['tam']/total*100:.1f}%)")
                    st.write(f"- English: {stats['eng']} segments ({stats['eng']/total*100:.1f}%)")
                
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
