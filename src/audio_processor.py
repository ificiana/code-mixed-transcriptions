"""
Audio Processor Module

This module handles the audio processing functionality for the Voice Isolation App.
It uses Facebook's denoiser for speech enhancement.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

from .utils.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


class AudioProcessor:
    """
    A class for processing audio files to enhance speech using Facebook's denoiser.

    Attributes:
        output_dir (Path): Directory to save processed audio files.
        model: Pretrained denoiser model.
        device (torch.device): Device to run model on (CPU/CUDA).
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the AudioProcessor.

        Args:
            output_dir: Optional directory to save processed audio files.
                       If None, a temporary directory will be used.
        """
        try:
            # Set output directory
            if output_dir is None:
                self.output_dir = Path(tempfile.mkdtemp(prefix="vocals_"))
            else:
                self.output_dir = Path(output_dir)

            # Ensure directory exists with proper permissions
            os.makedirs(self.output_dir, exist_ok=True)
            os.chmod(self.output_dir, 0o755)  # rwxr-xr-x permissions

            # Initialize device and model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = pretrained.dns64().to(self.device)

            logger.debug(f"Initialized output directory: {self.output_dir}")
            logger.debug(f"Using device: {self.device}")

        except Exception as e:
            logger.error(f"Error initializing AudioProcessor: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize AudioProcessor: {str(e)}")

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
            logger.debug(
                f"Audio file info - Samplerate: {info.samplerate}, Channels: {info.channels}, Format: {info.format}, Duration: {info.duration}s"
            )

            if info.duration < 0.1:  # Less than 100ms
                raise ValueError("Audio file is too short")

            if info.channels > 2:
                raise ValueError(
                    f"Audio file has unsupported number of channels: {info.channels}"
                )

        except Exception as e:
            raise ValueError(f"Failed to read audio file: {str(e)}")

    def process_audio(
        self,
        audio_path: str,
        chunk_size: int = 16000,  # Not used in new implementation
        overlap: float = 0.1,  # Not used in new implementation
    ) -> str:
        """
        Process an audio file to enhance speech and reduce noise.

        Args:
            audio_path: Path to the input audio file.
            chunk_size: Not used in new implementation.
            overlap: Not used in new implementation.

        Returns:
            Path to the processed audio file.
        """
        # Validate the audio file
        self._validate_audio_file(audio_path)

        try:
            # Load audio using torchaudio
            wav, sr = torchaudio.load(audio_path)
            logger.debug(f"Loaded audio - Shape: {wav.shape}, Sample rate: {sr}")

            # Move to device and convert audio
            wav = wav.to(self.device)
            wav = convert_audio(wav, sr, self.model.sample_rate, self.model.chin)

            # Process audio
            with torch.no_grad():
                denoised = self.model(wav[None])[0]

            # Create output filename
            input_filename = os.path.basename(audio_path)
            output_filename = f"enhanced_{os.path.splitext(input_filename)[0]}.wav"
            output_path = str(self.output_dir / output_filename)

            # Verify the output directory still exists
            if not self.output_dir.exists():
                os.makedirs(self.output_dir, exist_ok=True)
                logger.warning(
                    f"Output directory was missing, recreated: {self.output_dir}"
                )

            # Save processed audio using torchaudio
            # Ensure audio is in the correct format for saving (move to CPU and convert to float32)
            denoised = denoised.cpu().float()
            torchaudio.save(output_path, denoised, self.model.sample_rate)
            logger.info(f"Successfully saved enhanced audio to: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            raise

    def chunk_audio(self, audio_path: str, chunk_duration: int = 30) -> list[str]:
        """
        Split an audio file into chunks of specified duration.

        Args:
            audio_path: Path to the input audio file.
            chunk_duration: Duration of each chunk in seconds.

        Returns:
            List of paths to the chunked audio files.
        """
        try:
            # Load audio using torchaudio
            wav, sr = torchaudio.load(audio_path)

            # Calculate chunk size in samples
            chunk_size = int(chunk_duration * sr)

            # Calculate number of chunks
            num_chunks = int(torch.ceil(torch.tensor(wav.shape[1] / chunk_size)))

            chunk_paths = []

            # Create chunks
            for i in range(num_chunks):
                # Extract chunk
                start = i * chunk_size
                end = min(start + chunk_size, wav.shape[1])
                chunk = wav[:, start:end]

                # Create output filename
                input_filename = os.path.basename(audio_path)
                chunk_filename = (
                    f"chunk_{i+1}_{os.path.splitext(input_filename)[0]}.wav"
                )
                chunk_path = str(self.output_dir / chunk_filename)

                # Save chunk
                torchaudio.save(chunk_path, chunk, sr)
                chunk_paths.append(chunk_path)

            return chunk_paths

        except Exception as e:
            logger.error(f"Error chunking audio: {str(e)}", exc_info=True)
            raise

    def cleanup(self):
        """
        Clean up temporary files and resources.
        """
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Delete model
            if hasattr(self, "model"):
                del self.model

        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
