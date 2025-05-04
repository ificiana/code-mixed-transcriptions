"""Audio processing utilities."""

import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import streamlit as st
import torch
import torchaudio

from src.utils.file_mapping import FileMapper
from src.utils.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


def resample_audio(
    filepath: str,
    target_sr: int = 8000,
    cache_dir: Optional[Path] = None,
    cleanup: bool = True,
    file_mapper: Optional[FileMapper] = None,
    original_filename: Optional[str] = None,
) -> Tuple[str, int]:
    """Resample audio file to target sample rate if needed.

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate
        cache_dir: Directory to store resampled files (uses temp dir if None)
        cleanup: Whether to add resampled file to cleanup list
        file_mapper: Optional FileMapper to track resampled files

    Returns:
        Tuple of (output_path, sample_rate)
        If input SR matches target, returns original path
    """
    # Use original filename if provided, otherwise fallback to basename
    original_name = original_filename if original_filename else os.path.basename(filepath)

    # Check if we have a cached resampled version
    if file_mapper is not None:
        cached_path = file_mapper.get_mapping(original_name, f"resampled_{target_sr}")
        if cached_path and os.path.exists(cached_path):
            logger.info(f"Using cached {target_sr}Hz version")
            return cached_path, target_sr

    # Check if resampling is needed
    info = torchaudio.info(filepath)
    if info.sample_rate <= target_sr:
        return filepath, info.sample_rate

    # Load and resample
    st.info(f"Resampling audio to {target_sr}Hz for faster processing...")
    waveform, sr = torchaudio.load(filepath)
    resampler = torchaudio.transforms.Resample(sr, target_sr)

    # Use GPU if available
    if torch.cuda.is_available():
        waveform = waveform.cuda()
        resampler = resampler.cuda()

    # Resample
    waveform_resampled = resampler(waveform)

    # Move back to CPU if needed
    if torch.cuda.is_available():
        waveform_resampled = waveform_resampled.cpu()

    # Save resampled file
    if cache_dir is None:
        cache_dir = Path("temp_audio")
    cache_dir.mkdir(exist_ok=True)

    # Generate output path
    if file_mapper is not None:
        # Use hash-based name for cached files that includes original name for consistency
        file_hash = joblib.hash(original_name + "_" + waveform_resampled.numpy().tobytes().hex())
        output_path = str(cache_dir / f"{file_hash}_{target_sr}hz.wav")
    else:
        # Use original name for temporary files
        output_path = str(cache_dir / f"resampled_{target_sr}hz_{original_name}")

    # Save file
    torchaudio.save(output_path, waveform_resampled, target_sr)

    # Add to mapping if requested
    if file_mapper is not None:
        file_mapper.add_mapping(original_name, output_path, f"resampled_{target_sr}")

    # Add to cleanup if requested and not cached
    if cleanup and file_mapper is None and "temp_files" in st.session_state:
        st.session_state.temp_files.append(output_path)

    return output_path, target_sr
