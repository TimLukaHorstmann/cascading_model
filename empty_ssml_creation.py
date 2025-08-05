#!/usr/bin/env python3
"""
Empty SSML Creation - Programmatic step between text2breaks and breaks2ssml models

This module implements the missing "Empty SSML Creation (programmatic)" step shown in the cascade diagram.
It converts text with symbolic breaks (e.g., "Bonjour#250 comment vas-tu ?") into empty SSML templates
that can be fed to the breaks2ssml model.

Based on the format_z_ssml_template_from_parsed_sequence function from data_formatting_QwenB.py
"""

import re
from typing import List, Dict, Union


def create_empty_ssml_from_simple_breaks(text_with_breaks: str) -> str:
    """
    Convert text with simple <break/> tags to empty SSML template.
    
    This is the programmatic "Empty SSML Creation" step from the cascade diagram.
    This function follows EXACTLY the format_z_ssml_template_from_parsed_sequence format
    from data_formatting_QwenB.py to match the training data format.
    
    Args:
        text_with_breaks: Text with simple break tags (e.g., "Bonjour comment allez-vous ?<break/>")
        
    Returns:
        Empty SSML template with EXACT training data formatting
    
    Example:
        Input:  "Bonjour comment allez-vous ?<break/>"
        Output: " <prosody pitch=\"_%\" rate=\"_%\" volume=\"_%\">\n    Bonjour comment allez-vous ?\n  </prosody>\n  <break time=\"_ms\"/>"
    """
    if not text_with_breaks or not text_with_breaks.strip():
        return ""
    
    # Parse text with simple <break/> tags into segments (simulate parsed_sequence format)
    segments = parse_simple_breaks_to_parsed_sequence_format(text_with_breaks)
    
    # Apply the EXACT logic from format_z_ssml_template_from_parsed_sequence
    ssml_elements = []
    idx = 0
    
    while idx < len(segments):
        segment = segments[idx]
        stype = segment.get("type", "")
        text = segment.get("text", "")  # Text content
        
        if stype == "text":
            # For 'z', always include the prosody tag with placeholder attributes
            # EXACT format from training data: multiline with proper indentation
            ssml_elements.append(f'  <prosody pitch="_%\" rate="_%\" volume="_%\">\n    {text}\n  </prosody>')
            idx += 1
        elif stype == "break":
            current_breaks_tags = []
            temp_idx = idx
            while temp_idx < len(segments) and segments[temp_idx].get("type") == "break":
                # For 'z', always use the placeholder for time
                current_breaks_tags.append('<break time="_ms"/>')
                temp_idx += 1
            
            if current_breaks_tags:
                ssml_elements.append("  " + "".join(current_breaks_tags))
            idx = temp_idx
        else:
            # Fallback for any other content
            if text:
                ssml_elements.append(f"  {text}")
            idx += 1
    
    # Same joining logic as the original function
    final_z_output_parts = []
    num_elements = len(ssml_elements)
    
    for i, current_element_str in enumerate(ssml_elements):
        final_z_output_parts.append(current_element_str)
        is_break_element = "  <break" in current_element_str
        is_prosody_element = lambda s: s.startswith("  <prosody")
        
        if is_break_element:
            if (i + 1) < num_elements and is_prosody_element(ssml_elements[i+1]):
                final_z_output_parts.append("")  # Add empty string for extra newline
    
    if not final_z_output_parts: 
        return ""
    
    # EXACT format: starts with space and joins with newlines
    return " " + "\n".join(final_z_output_parts)


def parse_simple_breaks_to_parsed_sequence_format(text_with_breaks: str) -> List[Dict[str, str]]:
    """
    Parse text with simple <break/> tags into the parsed_sequence format used in training.
    
    This simulates the parsed_sequence structure from the training data to ensure
    we generate the exact same format as format_z_ssml_template_from_parsed_sequence.
    
    Args:
        text_with_breaks: Text with simple break tags (e.g., "Hello world<break/>more text")
        
    Returns:
        List of segments in parsed_sequence format with type and text fields
        
    Example:
        Input:  "Bonjour comment allez-vous ?<break/>"
        Output: [
            {'type': 'text', 'text': 'Bonjour comment allez-vous ?'},
            {'type': 'break'}
        ]
    """
    segments = []
    
    # Split on <break/> tags
    parts = re.split(r'<break\s*/>', text_with_breaks)
    
    for i, part in enumerate(parts):
        # Add text segment if not empty
        if part.strip():
            segments.append({
                'type': 'text',
                'text': part.strip()
            })
        
        # Add break segment after each part except the last
        if i < len(parts) - 1:
            segments.append({
                'type': 'break'
            })
    
    return segments


def create_empty_ssml_multiline(text_with_breaks: str) -> str:
    """
    Create empty SSML template with multiline formatting (same as main function).
    
    This is just an alias to the main function since we now follow the exact training format.
    
    Args:
        text_with_breaks: Text with simple break tags
        
    Returns:
        Empty SSML template with exact training data formatting
    """
    return create_empty_ssml_from_simple_breaks(text_with_breaks)


# For backward compatibility and testing
def test_empty_ssml_creation():
    """Test the empty SSML creation functions"""
    test_cases = [
        "Bonjour comment allez-vous ?<break/>",
        "Bonjour je m'appelle Bertrand Perier.<break/>Je suis avocat à la cour.",
        "Hello world<break/>this is a test<break/>",
        "Simple text without breaks",
        "<break/>starts with break",
        "ends with break<break/>",
        "Bonjour comment allez-vous ?",  # No breaks
        ""
    ]
    
    print("Testing Empty SSML Creation (EXACT training format)")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{test_input}'")
        
        # Test main function with exact training format
        result = create_empty_ssml_from_simple_breaks(test_input)
        print(f"   Output:")
        if result:
            # Print with visible formatting
            print(f"'{result}'")
            print(f"   Formatted output:")
            print(result)
        else:
            print(f"   (empty)")
        
        # Test parsing
        segments = parse_simple_breaks_to_parsed_sequence_format(test_input)
        print(f"   Parsed segments: {segments}")


if __name__ == "__main__":
    test_empty_ssml_creation()
