"""
French SSML Cascade Models

This package provides inference for French SSML cascade models that improve
synthetic speech quality through intelligent prosody control.

Main classes:
    - Text2BreaksInference: Converts text to text with breaks
    - Breaks2SSMLInference: Converts breaks to SSML
    - CascadedInference: Full text-to-SSML pipeline (memory efficient)
    - SharedModelManager: Manages shared base model with swappable adapters
"""

from .text2breaks_inference import Text2BreaksInference
from .breaks2ssml_inference import Breaks2SSMLInference, CascadedInference
from .empty_ssml_creation import create_empty_ssml_from_simple_breaks
from .shared_models import SharedModelManager, get_shared_manager

__version__ = "0.1.0"
__all__ = [
    "Text2BreaksInference",
    "Breaks2SSMLInference", 
    "CascadedInference",
    "create_empty_ssml_from_simple_breaks",
    "SharedModelManager",
    "get_shared_manager"
]
