#!/usr/bin/env python
"""
Setup script for Voice Isolation App.
This script creates a virtual environment and installs the required dependencies.
"""

import argparse
import os
import platform
import subprocess
import sys


def create_venv():
    """Create a virtual environment."""
    print("Creating virtual environment...")

    # Check if venv already exists
    if os.path.exists("venv"):
        print("Virtual environment already exists.")
        return True

    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error creating virtual environment: {e}")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")

    # Get the path to the Python executable in the virtual environment
    if platform.system() == "Windows":
        python_path = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join("venv", "bin", "python")

    if not os.path.exists(python_path):
        print(f"Error: Python executable not found at {python_path}")
        return False

    try:
        # Install dependencies
        subprocess.run(
            [python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True
        )
        print("Dependencies installed successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def main():
    """Main function to run the setup."""
    parser = argparse.ArgumentParser(description="Setup Voice Isolation App")
    parser.add_argument(
        "--skip-venv", action="store_true", help="Skip virtual environment creation"
    )

    args = parser.parse_args()

    print("Voice Isolation App Setup")
    print("========================")

    # Create virtual environment
    if not args.skip_venv:
        if not create_venv():
            return 1

    # Install dependencies
    if not install_dependencies():
        return 1

    print("\nSetup completed successfully!")
    print("\nTo run the Voice Isolation App:")

    if platform.system() == "Windows":
        print("  venv\\Scripts\\python.exe run.py")
    else:
        print("  venv/bin/python run.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
