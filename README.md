# Audio Processing App

A comprehensive Streamlit application for audio processing, including chunking, voice isolation, speaker diarization, and transcription.

## Overview

This application provides a suite of tools for working with audio files, particularly focused on speech processing. It helps you split long recordings, clean audio by isolating vocals, identify different speakers, and transcribe Tamil-English code-mixed speech.

## Features

- **Audio Chunking**
  - Split long audio files into smaller chunks
  - Customize chunk duration
  - Download individual chunks

- **Voice Isolation**
  - Isolate vocals using deep learning
  - Remove background noise
  - Multiple model options for different quality/speed tradeoffs
  - Visualize audio waveforms and spectrograms
  - Compare original and processed audio

- **Speaker Diarization**
  - Identify and segment different speakers
  - Generate RTTM files with speaker timings
  - Visualize speaker segments

- **Audio Transcription**
  - Transcribe Tamil-English code-mixed speech
  - Support for non-native English accents
  - Speaker-wise transcription with timestamps
  - Language statistics for each speaker

## Quick Start

### Setup with Virtual Environment

1. Run the setup script to create a virtual environment and install dependencies:
   ```bash
   python setup.py
   ```

2. Run the application:
   ```bash
   # On Windows
   venv\Scripts\python app.py
   
   # On macOS/Linux
   venv/bin/python app.py
   ```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   # Directly with Python
   python app.py
   
   # Or with Streamlit
   streamlit run app.py
   ```

5. Open your browser at http://localhost:8501

## Technical Stack

- **Streamlit**: Web application framework
- **Demucs**: Deep learning model for audio source separation
- **PyTorch/Torchaudio**: Audio processing
- **Librosa**: Audio analysis and visualization
- **Pyannote.audio**: Speaker diarization
- **Transformers**: Speech recognition and transcription
- **Facebook MMS**: Multilingual speech model for Tamil-English transcription

## Project Structure

- `app.py`: Main application entry point
- `pages/`: Individual pages for each feature
  - `1_audio_chunking.py`: Audio chunking functionality
  - `2_voice_isolation.py`: Voice isolation functionality
  - `3_speaker_diarization.py`: Speaker diarization functionality
  - `4_transcription.py`: Transcription functionality
- `src/`: Source code for the application
  - `utils/`: Utility functions for audio processing, visualization, etc.
- `requirements.txt`: Dependencies for the application
- `setup.py`: Setup script for creating a virtual environment and installing dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.
