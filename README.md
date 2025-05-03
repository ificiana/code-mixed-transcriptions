# Voice Isolation App

A simple Streamlit application for audio denoising and voice isolation.

## Overview

This application helps you clean audio files by isolating vocals and removing background noise. It uses the Demucs deep learning model to separate vocals from other audio components.

## Features

- Upload audio files (MP3, WAV, OGG, FLAC)
- Isolate vocals using deep learning
- Visualize audio waveforms and spectrograms
- Compare original and processed audio
- Download the isolated vocals

## Quick Start

### Setup with Virtual Environment

1. Run the setup script to create a virtual environment and install dependencies:
   ```bash
   python setup.py
   ```

2. Run the application:
   ```bash
   # On Windows
   venv\Scripts\python run.py
   
   # On macOS/Linux
   venv/bin/python run.py
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
   # With the run script
   python run.py
   
   # Directly with Streamlit
   streamlit run app/app.py
   ```

5. Open your browser at http://localhost:8501

## Technical Stack

- Streamlit: Web application framework
- Demucs: Deep learning model for audio source separation
- PyTorch/Torchaudio: Audio processing
- Librosa: Audio analysis and visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.
