"""
Download and save pyannote models locally.
This script requires you to be logged in via huggingface-cli:

1. Install huggingface-cli:
   pip install --upgrade huggingface-hub

2. Login to Hugging Face:
   huggingface-cli login

3. Accept the user conditions at:
   https://huggingface.co/pyannote/speaker-diarization-3.1
"""

from pathlib import Path

from pyannote.audio import Pipeline


def download_models():
    """Download and cache pyannote models locally"""

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Download diarization pipeline
    print("Downloading speaker diarization pipeline...")
    try:
        # Initialize pipeline to download and cache models
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("Successfully downloaded and cached pipeline")

    except Exception as e:
        print(f"Error downloading diarization pipeline: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed huggingface-cli: pip install --upgrade huggingface-hub")
        print("2. Logged in: huggingface-cli login")
        print(
            "3. Accepted the user conditions at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        return

    print("\nAll models downloaded and cached successfully!")


if __name__ == "__main__":
    download_models()
