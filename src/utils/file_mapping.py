"""File mapping utilities for tracking original and processed files."""

import json
from pathlib import Path
from typing import Dict, Optional


class FileMapper:
    """Manages mapping between original and processed files."""

    def __init__(self, directory: Path):
        """Initialize mapper with directory for map file.

        Args:
            directory: Directory to store the mapping file
        """
        self.directory = directory
        self.map_file = directory / "file_mapping.json"
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> Dict:
        """Load existing mapping or create new one."""
        if self.map_file.exists():
            with open(self.map_file, "r") as f:
                return json.load(f)
        return {}

    def _save_mapping(self):
        """Save current mapping to file."""
        with open(self.map_file, "w") as f:
            json.dump(self.mapping, f, indent=2)

    def add_mapping(self, original_file: str, processed_file: str, file_type: str):
        """Add a new file mapping.

        Args:
            original_file: Original filename
            processed_file: Processed/cached filename
            file_type: Type of mapping (e.g., 'cache', 'rttm')
        """
        if original_file not in self.mapping:
            self.mapping[original_file] = {}
        self.mapping[original_file][file_type] = processed_file
        self._save_mapping()

    def get_mapping(self, original_file: str, file_type: str) -> Optional[str]:
        """Get processed file path for original file.

        Args:
            original_file: Original filename
            file_type: Type of mapping to retrieve

        Returns:
            Path to processed file or None if not found
        """
        return self.mapping.get(original_file, {}).get(file_type)

    def get_original_file(self, processed_file: str, file_type: str) -> Optional[str]:
        """Get original filename from processed file.

        Args:
            processed_file: Processed/cached filename
            file_type: Type of mapping to search

        Returns:
            Original filename or None if not found
        """
        for orig, mappings in self.mapping.items():
            if mappings.get(file_type) == processed_file:
                return orig
        return None

    def remove_mapping(self, original_file: str, file_type: str):
        """Remove a specific file mapping.

        Args:
            original_file: Original filename
            file_type: Type of mapping to remove
        """
        if original_file in self.mapping and file_type in self.mapping[original_file]:
            del self.mapping[original_file][file_type]
            if not self.mapping[original_file]:  # If no mappings left for this file
                del self.mapping[original_file]
            self._save_mapping()
            
    def clear_mappings(self, file_type: str):
        """Clear all mappings of a specific type.
        
        Args:
            file_type: Type of mappings to clear (e.g., 'cache', 'rttm')
        """
        # Create list of files to remove to avoid modifying dict during iteration
        to_remove = []
        for original_file in self.mapping:
            if file_type in self.mapping[original_file]:
                to_remove.append(original_file)
        
        # Remove mappings
        for original_file in to_remove:
            self.remove_mapping(original_file, file_type)
