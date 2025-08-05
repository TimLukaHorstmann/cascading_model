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
    
    Args:
        text_with_breaks: Text with simple break tags (e.g., "Bonjour comment allez-vous ?<break/>")
        
    Returns:
        Empty SSML template (e.g., "<prosody pitch=\"_%\" rate=\"_%\" volume=\"_%\">Bonjour comment allez-vous ?</prosody><break time=\"_ms\"/>")
    
    Example:
        Input:  "Bonjour comment allez-vous ?<break/>"
        Output: "<prosody pitch=\"_%\" rate=\"_%\" volume=\"_%\">Bonjour comment allez-vous ?</prosody><break time=\"_ms\"/>"
    """
    if not text_with_breaks or not text_with_breaks.strip():
        return ""
    
    # Parse text with simple <break/> tags into segments
    segments = parse_simple_breaks(text_with_breaks)
    
    # Convert segments to empty SSML template
    ssml_parts = []
    
    for segment in segments:
        if segment['type'] == 'text':
            text = segment['text'].strip()
            if text:
                # Wrap text in empty prosody tags with placeholder attributes
                ssml_parts.append(f'<prosody pitch="_%\" rate="_%\" volume="_%\">{text}</prosody>')
        elif segment['type'] == 'break':
            # Convert simple break to empty break tag with placeholder time
            ssml_parts.append('<break time="_ms"/>')
    
    return ''.join(ssml_parts)


def parse_simple_breaks(text_with_breaks: str) -> List[Dict[str, str]]:
    """
    Parse text with simple <break/> tags into structured segments.
    
    Args:
        text_with_breaks: Text with simple break tags (e.g., "Hello world<break/>more text")
        
    Returns:
        List of segments with type and content
        
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
    Create empty SSML template with multiline formatting (similar to training data format).
    
    Args:
        text_with_breaks: Text with simple break tags (e.g., "Bonjour<break/>comment vas-tu ?")
        
    Returns:
        Multiline empty SSML template
        
    Example:
        Input:  "Bonjour comment allez-vous ?<break/>"
        Output: 
        " <prosody pitch=\"_%\" rate=\"_%\" volume=\"_%\">
           Bonjour comment allez-vous ?
         </prosody>
         <break time=\"_ms\"/>"
    """
    if not text_with_breaks or not text_with_breaks.strip():
        return ""
    
    segments = parse_simple_breaks(text_with_breaks)
    ssml_elements = []
    
    for segment in segments:
        if segment['type'] == 'text':
            text = segment['text'].strip()
            if text:
                # Multiline prosody format
                ssml_elements.append(f'  <prosody pitch="_%\" rate="_%\" volume="_%\">\n    {text}\n  </prosody>')
        elif segment['type'] == 'break':
            # Break tag
            ssml_elements.append('  <break time="_ms"/>')
    
    # Join with newlines and add spacing between breaks and prosody
    final_parts = []
    for i, element in enumerate(ssml_elements):
        final_parts.append(element)
        
        # Add extra newline between break and following prosody
        is_break = '<break' in element
        if is_break and i + 1 < len(ssml_elements) and '<prosody' in ssml_elements[i + 1]:
            final_parts.append('')  # Empty string adds extra newline
    
    if final_parts:
        return ' ' + '\n'.join(final_parts)
    return ""


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
    
    print("Testing Empty SSML Creation")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{test_input}'")
        
        # Test single line format
        result_single = create_empty_ssml_from_simple_breaks(test_input)
        print(f"   Single: {result_single}")
        
        # Test multiline format  
        result_multi = create_empty_ssml_multiline(test_input)
        if result_multi:
            print(f"   Multi:\n{result_multi}")
        else:
            print(f"   Multi: (empty)")
        
        # Test parsing
        segments = parse_simple_breaks(test_input)
        print(f"   Segments: {segments}")


if __name__ == "__main__":
    test_empty_ssml_creation()
