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
from .utils.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

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
        
        try:
            # Set output directory
            if output_dir is None:
                self.output_dir = Path(tempfile.mkdtemp(prefix="vocals_"))
            else:
                self.output_dir = Path(output_dir)
            
            # Ensure directory exists with proper permissions
            os.makedirs(self.output_dir, exist_ok=True)
            os.chmod(self.output_dir, 0o755)  # rwxr-xr-x permissions
            
            logger.debug(f"Initialized output directory: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error initializing output directory: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize output directory: {str(e)}")
    
    def _load_model(self, model_name: str) -> Any:
        """
        Load a Demucs model if not already loaded.
        
        Args:
            model_name: Name of the Demucs model to load.
            
        Returns:
            The loaded model.
        """
        # Import required modules
        _, _, get_model, _, _ = _import_torch_modules()
        
        if model_name not in self.models:
            logger.info(f"Loading model: {model_name}")
            model = get_model(model_name)
            model.to(self.device)
            self.models[model_name] = model
        return self.models[model_name]
    
    def _validate_audio_file(self, audio_path: str) -> None:
        """
        Validate the audio file before processing.
        
        Args:
            audio_path: Path to the audio file to validate.
            
        Raises:
            ValueError: If the file is invalid or cannot be read.
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
            
        try:
            info = sf.info(audio_path)
            logger.debug(f"Audio file info - Samplerate: {info.samplerate}, Channels: {info.channels}, Format: {info.format}, Duration: {info.duration}s")
            
            if info.duration < 0.1:  # Less than 100ms
                raise ValueError("Audio file is too short")
                
            if info.channels > 2:
                raise ValueError(f"Audio file has unsupported number of channels: {info.channels}")
                
        except Exception as e:
            raise ValueError(f"Failed to read audio file: {str(e)}")
    
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
        # Validate the audio file
        self._validate_audio_file(audio_path)
        
        # Set up PyTorch and import required modules
        torch, torchaudio, get_model, apply_model, load_track = _import_torch_modules()
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"Using device: {self.device}")
        logger.debug("PyTorch modules imported successfully")
        
        # Load the model
        model = self._load_model(model_name)
        logger.debug(f"Model loaded successfully: {model_name}")
        
        try:
            # Load the audio track
            wav = load_track(audio_path, model.audio_channels, model.samplerate)
            logger.debug(f"Audio loaded - Type: {type(wav)}, Shape: {wav.shape}, dtype: {wav.dtype}")
            
            # Move to correct device and ensure correct dtype
            if not isinstance(wav, torch.Tensor):
                wav = torch.from_numpy(wav)
            wav = wav.to(device=self.device, dtype=torch.float32)
            logger.debug(f"Converted to tensor - Type: {type(wav)}, Shape: {wav.shape}, dtype: {wav.dtype}, device: {wav.device}")
            
            # Get the source names from the model
            sources = model.sources
            logger.debug(f"Model sources: {sources}")
            
            # Apply the model to separate sources
            ref = wav.mean(0)  # Downmix to mono for reference
            logger.debug(f"Reference mean - Type: {type(ref)}, Shape: {ref.shape}")
            wav = (wav - ref.mean()) / ref.std()  # Normalize
            logger.debug(f"After normalization - Type: {type(wav)}, Shape: {wav.shape}")
        
            # Apply the model
            with torch.no_grad():
                logger.debug(f"Input to apply_model - Type: {type(wav[None])}, Shape: {wav[None].shape}")
                sources = apply_model(
                    model, 
                    wav[None], 
                    device=self.device, 
                    shifts=shifts, 
                    split=True, 
                    overlap=overlap,
                    segment=segment
                )[0]
                logger.debug(f"After apply_model - Type: {type(sources)}, Shape: {sources.shape}")
            
            # Convert back to numpy
            sources = sources.cpu().numpy()
            logger.debug(f"Converted to numpy - Type: {type(sources)}, Shape: {sources.shape}")
            
            # Get the vocals index
            vocals_idx = model.sources.index("vocals")
            logger.debug(f"Vocals index: {vocals_idx}")
            
            # Extract vocals
            vocals = sources[vocals_idx]
            logger.debug(f"Extracted vocals - Type: {type(vocals)}, Shape: {vocals.shape}")
            
        except Exception as e:
            logger.error(f"Error during audio processing: {str(e)}", exc_info=True)
            raise
        
        # Create output filename
        input_filename = os.path.basename(audio_path)
        output_filename = f"vocals_{os.path.splitext(input_filename)[0]}.wav"
        output_path = str(self.output_dir / output_filename)
        
        try:
            # Verify the output directory still exists
            if not self.output_dir.exists():
                os.makedirs(self.output_dir, exist_ok=True)
                logger.warning(f"Output directory was missing, recreated: {self.output_dir}")
            
            # Convert vocals to numpy and ensure correct shape for soundfile
            vocals_numpy = vocals
            if vocals_numpy.shape[0] > vocals_numpy.shape[1]:
                vocals_numpy = vocals_numpy.T  # Transpose if needed
            
            logger.debug(f"Vocals array for saving - Type: {type(vocals_numpy)}, Shape: {vocals_numpy.shape}")
            
            # Normalize audio to prevent clipping
            max_val = np.abs(vocals_numpy).max()
            if max_val > 1.0:
                vocals_numpy = vocals_numpy / max_val
                logger.debug(f"Normalized audio (max value was: {max_val})")
            
            # Save using soundfile
            try:
                sf.write(
                    output_path,
                    vocals_numpy.T,  # soundfile expects (frames, channels)
                    model.samplerate,
                    format='WAV',
                    subtype='PCM_16'  # Use 16-bit PCM format
                )
            except Exception as e:
                logger.error(f"Failed to save audio with soundfile: {str(e)}")
                # Try alternate save method
                try:
                    import scipy.io.wavfile as wavfile
                    logger.debug("Attempting to save with scipy.io.wavfile")
                    # Convert to 16-bit PCM
                    vocals_int16 = (vocals_numpy * 32767).astype(np.int16)
                    wavfile.write(output_path, model.samplerate, vocals_int16.T)
                except Exception as e2:
                    logger.error(f"Failed to save audio with scipy.io.wavfile: {str(e2)}")
                    raise RuntimeError(f"Could not save audio file using multiple methods: {str(e)}, {str(e2)}")
            logger.info(f"Successfully saved vocals to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving audio file: {str(e)}", exc_info=True)
            raise
    
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
