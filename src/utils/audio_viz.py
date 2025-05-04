"""
Audio Visualization Utilities

This module provides utilities for generating audio visualizations (waveforms and spectrograms)
in parallel using ThreadPoolExecutor.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def generate_waveform(y: np.ndarray, sr: int, title: str) -> plt.Figure:
    """
    Generate waveform plot.

    Args:
        y: Audio time series
        sr: Sampling rate
        title: Plot title

    Returns:
        Matplotlib figure containing the waveform plot
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    return fig


def generate_spectrogram(y: np.ndarray, sr: int, title: str) -> plt.Figure:
    """
    Generate spectrogram plot.

    Args:
        y: Audio time series
        sr: Sampling rate
        title: Plot title

    Returns:
        Matplotlib figure containing the spectrogram plot
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis="log", x_axis="time", sr=sr, ax=ax)
    ax.set_title(title)
    return fig


def generate_visualizations(
    y: np.ndarray,
    sr: int,
    title: str,
    include_waveform: bool = True,
    include_spectrogram: bool = True,
) -> Dict[str, plt.Figure]:
    """
    Generate audio visualizations in parallel.

    Args:
        y: Audio time series
        sr: Sampling rate
        title: Base title for plots
        include_waveform: Whether to generate waveform plot
        include_spectrogram: Whether to generate spectrogram plot

    Returns:
        Dictionary containing the generated plots:
        {
            'waveform': waveform figure (if requested),
            'spectrogram': spectrogram figure (if requested)
        }
    """
    plots = {}

    with ThreadPoolExecutor() as executor:
        futures = {}

        if include_waveform:
            futures["waveform"] = executor.submit(
                generate_waveform, y, sr, f"{title} Waveform"
            )

        if include_spectrogram:
            futures["spectrogram"] = executor.submit(
                generate_spectrogram, y, sr, f"{title} Spectrogram"
            )

        # Collect results
        for name, future in futures.items():
            plots[name] = future.result()

    return plots


def display_audio_visualizations(
    y: np.ndarray,
    sr: int,
    title: str,
    audio_data: bytes = None,
    audio_format: str = "audio/wav",
) -> None:
    """
    Generate and display audio visualizations in Streamlit.

    Args:
        y: Audio time series
        sr: Sampling rate
        title: Title for the visualization section
        audio_data: Optional audio data for playback
        audio_format: Audio format for playback
    """
    # Generate visualizations in parallel
    plots = generate_visualizations(y, sr, title)

    # Display waveform and audio player
    st.markdown(f"**{title}**")
    if "waveform" in plots:
        st.pyplot(plots["waveform"])

    # Display audio player if data provided
    if audio_data is not None:
        st.audio(audio_data, format=audio_format)

    # Display spectrogram
    if "spectrogram" in plots:
        st.pyplot(plots["spectrogram"])
