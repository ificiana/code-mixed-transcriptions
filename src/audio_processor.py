"""
Audio Processor Module

This module handles the audio processing functionality for the Voice Isolation App.
It uses Demucs to separate vocals from the audio.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Any, List
import librosa
import soundfile as sf
import numpy as np

# Safely import torch-related modules
def _import_torch_modules():
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.separate import load_track
        return torch, torchaudio, get_model, apply_model, load_track
    except ImportError as e:
        raise ImportError(f"Required modules not found: {str(e)}")


class AudioProcessor:
    """
    A class for processing audio files to isolate vocals using Demucs.
    
    Attributes:
        device (torch.device): The device to run the model on (CPU or CUDA).
        models (Dict[str, Any]): A dictionary of loaded models.
        output_dir (Path): Directory to save processed audio files.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the AudioProcessor.
        
        Args:
            output_dir: Optional directory to save processed audio files.
                        If None, a temporary directory will be used.
        """
        # Initialize device attribute
        self.device = None
        
        # Initialize models dictionary
        self.models = {}
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp())
        else:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_model(self, model_name: str) -> Any:
        """
        Load a Demucs model if not already loaded.
        
        Args:
            model_name: Name of the Demucs model to load.
            
        Returns:
            The loaded model.
        """
        if model_name not in self.models:
            print(f"Loading model: {model_name}")
            model = get_model(model_name)
            model.to(self.device)
            self.models[model_name] = model
        return self.models[model_name]
    
    def _setup_torch(self):
        """Set up PyTorch and related imports."""
        torch, torchaudio, _, _, _ = _import_torch_modules()
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
        return torch, torchaudio

    def process_audio(
        self, 
        audio_path: str, 
        model_name: str = "htdemucs",
        shifts: int = 2, 
        overlap: float = 0.25,
        segment: Optional[int] = None
    ) -> str:
        """
        Process an audio file to isolate vocals.
        
        Args:
            audio_path: Path to the input audio file.
            model_name: Name of the Demucs model to use.
            shifts: Number of random shifts for augmenting separation.
            overlap: Overlap between the splits.
            segment: Segment size to use for splitting audio.
                    If None, use the default from the model.
        
        Returns:
            Path to the processed vocals file.
        """
        # Import required modules
        torch, torchaudio, get_model, apply_model, load_track = _import_torch_modules()
        
        # Load the model
        if model_name not in self.models:
            print(f"Loading model: {model_name}")
            model = get_model(model_name)
            model.to(self.device)
            self.models[model_name] = model
        model = self.models[model_name]
        
        # Load the audio track
        wav = load_track(audio_path, model.audio_channels, model.samplerate)
        
        # Get the source names from the model
        sources = model.sources
        
        # Apply the model to separate sources
        ref = wav.mean(0)  # Downmix to mono for reference
        wav = (wav - ref.mean()) / ref.std()  # Normalize
        
        # Import torch and set up device
        torch, torchaudio = self._setup_torch()
        
        # Convert to tensor using recommended approach
        wav = torch.from_numpy(wav).to(dtype=torch.float32)
        
        # Apply the model
        with torch.no_grad():
            sources = apply_model(
                model, 
                wav[None], 
                device=self.device, 
                shifts=shifts, 
                split=True, 
                overlap=overlap,
                segment=segment
            )[0]
        
        # Convert back to numpy
        sources = sources.cpu().numpy()
        
        # Get the vocals index
        vocals_idx = model.sources.index("vocals")
        
        # Extract vocals
        vocals = sources[vocals_idx]
        
        # Create output filename
        input_filename = os.path.basename(audio_path)
        output_filename = f"vocals_{os.path.splitext(input_filename)[0]}.wav"
        output_path = str(self.output_dir / output_filename)
        
        # Save the vocals to a file
        torchaudio.save(
            output_path,
            torch.from_numpy(vocals).to(dtype=torch.float32).t(),
            model.samplerate
        )
        
        return output_path
    
    def chunk_audio(self, audio_path: str, chunk_duration: int = 30) -> List[str]:
        """
        Split an audio file into chunks of specified duration.
        
        Args:
            audio_path: Path to the input audio file.
            chunk_duration: Duration of each chunk in seconds.
            
        Returns:
            List of paths to the chunked audio files.
        """
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate chunk size in samples
        chunk_size = int(chunk_duration * sr)
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(len(y) / chunk_size))
        
        chunk_paths = []
        
        # Create chunks
        for i in range(num_chunks):
            # Extract chunk
            start = i * chunk_size
            end = min(start + chunk_size, len(y))
            chunk = y[start:end]
            
            # Create output filename
            input_filename = os.path.basename(audio_path)
            chunk_filename = f"chunk_{i+1}_{os.path.splitext(input_filename)[0]}.wav"
            chunk_path = str(self.output_dir / chunk_filename)
            
            # Save chunk
            sf.write(chunk_path, chunk, sr)
            chunk_paths.append(chunk_path)
        
        return chunk_paths

    def cleanup(self):
        """
        Clean up temporary files and resources.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
