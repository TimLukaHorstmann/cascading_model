#!/usr/bin/env python3
"""
Comprehensive demo script for the French SSML cascade models

This script demonstrates:
1. Text-to-Breaks prediction (hi-paris/ssml-text2breaks-fr-lora)
2. Breaks-to-SSML conversion (hi-paris/ssml-breaks2ssml-fr-lora)
3. Full cascade from plain text to SSML

Usage:
    python demo.py                    # Run with default examples
    python demo.py --interactive      # Interactive mode
    python demo.py --text "Your text here"
"""

import torch
from text2breaks_inference import Text2BreaksInference
from breaks2ssml_inference import Breaks2SSMLInference, CascadedInference
import argparse
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_examples():
    """Run demo with predefined examples"""
    
    # Example texts
    examples = [
        "Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour.",
        "Comment allez-vous aujourd'hui ? J'espère que tout va bien pour vous.",
        "Il était une fois, dans un pays lointain, un roi très sage qui gouvernait son royaume avec bienveillance.",
        "Nous vous remercions de votre attention. Avez-vous des questions à poser maintenant ?",
        "Le temps est magnifique aujourd'hui. Voulez-vous faire une promenade dans le parc ?"
    ]
    
    print("=" * 80)
    print("🗣️  French SSML Cascade Models Demo")
    print("=" * 80)
    print()
    
    # Initialize models
    print("🔧 Initializing models...")
    start_time = time.time()
    
    try:
        # Initialize individual models
        text2breaks = Text2BreaksInference()
        breaks2ssml = Breaks2SSMLInference()
        
        # Initialize cascade
        cascade = CascadedInference()
        
        init_time = time.time() - start_time
        print(f"✅ Models loaded successfully in {init_time:.1f} seconds\n")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Run examples
    for i, text in enumerate(examples, 1):
        print(f"📝 Example {i}/{len(examples)}")
        print(f"Input: {text}")
        print()
        
        try:
            # Step 1: Text to breaks
            print("🔹 Step 1: Adding symbolic breaks...")
            start_time = time.time()
            text_with_breaks = text2breaks.predict(text, temperature=0.7)
            step1_time = time.time() - start_time
            print(f"   Result: {text_with_breaks}")
            print(f"   Time: {step1_time:.2f}s")
            print()
            
            # Step 2: Breaks to SSML
            print("🔹 Step 2: Converting to SSML...")
            start_time = time.time()
            ssml_output = breaks2ssml.predict(text_with_breaks, temperature=0.3)
            step2_time = time.time() - start_time
            print(f"   Result: {ssml_output}")
            print(f"   Time: {step2_time:.2f}s")
            print()
            
            total_time = step1_time + step2_time
            print(f"⏱️  Total processing time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")
        
        print("-" * 80)
        print()


def interactive_mode():
    """Run in interactive mode"""
    
    print("=" * 80)
    print("🗣️  Interactive French SSML Cascade")
    print("=" * 80)
    print()
    print("Choose mode:")
    print("1. Full cascade (text → breaks → SSML)")
    print("2. Text to breaks only")
    print("3. Breaks to SSML only")
    print()
    
    try:
        mode = input("Select mode (1-3): ").strip()
    except KeyboardInterrupt:
        return
    
    print("\n🔧 Initializing models...")
    
    try:
        if mode == "1":
            inferencer = CascadedInference()
            input_prompt = "Enter French text"
        elif mode == "2":
            inferencer = Text2BreaksInference()
            input_prompt = "Enter French text"
        elif mode == "3":
            inferencer = Breaks2SSMLInference()
            input_prompt = "Enter text with symbolic breaks (e.g., 'Bonjour#250 comment allez-vous ?')"
        else:
            print("Invalid mode selected.")
            return
            
        print("✅ Models loaded successfully!")
        print(f"\n{input_prompt} (empty line to exit):")
        
        while True:
            try:
                text = input("\n> ").strip()
                if not text:
                    break
                    
                start_time = time.time()
                result = inferencer.predict(text, temperature=0.5)
                process_time = time.time() - start_time
                
                print(f"Output: {result}")
                print(f"Time: {process_time:.2f}s")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"❌ Error initializing models: {e}")


def process_single_text(text, mode="cascade"):
    """Process a single text input"""
    
    print(f"🔧 Initializing models for {mode} mode...")
    
    try:
        if mode == "cascade":
            inferencer = CascadedInference()
        elif mode == "text2breaks":
            inferencer = Text2BreaksInference()
        elif mode == "breaks2ssml":
            inferencer = Breaks2SSMLInference()
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        print("✅ Models loaded successfully!")
        print()
        
        print(f"Input:  {text}")
        
        start_time = time.time()
        result = inferencer.predict(text, temperature=0.5)
        process_time = time.time() - start_time
        
        print(f"Output: {result}")
        print(f"Time: {process_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="French SSML Cascade Models Demo")
    parser.add_argument("--text", help="Input text to process")
    parser.add_argument("--mode", choices=["cascade", "text2breaks", "breaks2ssml"], 
                       default="cascade", help="Processing mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--examples", action="store_true", help="Run with example texts")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.text:
        process_single_text(args.text, args.mode)
    elif args.examples:
        demo_examples()
    else:
        # Default: run examples
        demo_examples()


if __name__ == "__main__":
    main()
