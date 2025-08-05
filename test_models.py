#!/usr/bin/env python3
"""
Simple test script to verify the models work correctly.
Run this after installation to check if everything is set up properly.
"""

import sys
import traceback
from text2breaks_inference import Text2BreaksInference
from breaks2ssml_inference import Breaks2SSMLInference, CascadedInference

def test_imports():
    """Test if all required packages are available"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"   ✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"   ❌ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"   ✅ PEFT {peft.__version__}")
    except ImportError as e:
        print(f"   ❌ PEFT: {e}")
        return False
    
    print("   ✅ All imports successful!")
    return True


def test_model_loading():
    """Test if models can be loaded"""
    print("\n🔧 Testing model loading...")
    
    try:
        print("   Loading text2breaks model...")
        text2breaks = Text2BreaksInference()
        print("   ✅ Text2breaks model loaded")
        
        print("   Loading breaks2ssml model...")
        breaks2ssml = Breaks2SSMLInference()
        print("   ✅ Breaks2ssml model loaded")
        
        print("   ✅ All models loaded successfully!")
        return text2breaks, breaks2ssml
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        traceback.print_exc()
        return None, None


def test_inference(text2breaks, breaks2ssml):
    """Test inference with a simple example"""
    print("\n🧪 Testing inference...")
    
    test_text = "Bonjour comment allez-vous ?"
    print(f"   Input: {test_text}")
    
    try:
        print("   Testing text2breaks...")
        text_with_breaks = text2breaks.predict(test_text, temperature=0.5)
        print(f"   Step 1 result: {text_with_breaks}")
        
        print("   Testing breaks2ssml...")
        ssml_result = breaks2ssml.predict(text_with_breaks, temperature=0.3)
        print(f"   Step 2 result: {ssml_result}")
        
        print("   ✅ Inference test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Inference test failed: {e}")
        traceback.print_exc()
        return False


def test_cascade():
    """Test the full cascade"""
    print("\n🔗 Testing full cascade...")
    
    test_text = "Bonsoir comment ça va ?"
    print(f"   Input: {test_text}")
    
    try:
        cascade = CascadedInference()
        result = cascade.predict(test_text)
        print(f"   Cascade result: {result}")
        print("   ✅ Cascade test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Cascade test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("🧪 French SSML Models - Test Suite")
    print("=" * 60)
    
    if not test_imports():
        print("\n❌ Import test failed. Please install required packages.")
        return False
    text2breaks, breaks2ssml = test_model_loading()
    if text2breaks is None or breaks2ssml is None:
        print("\n❌ Model loading failed. Please check your internet connection and HuggingFace access.")
        return False
    
    if not test_inference(text2breaks, breaks2ssml):
        print("\n❌ Inference test failed.")
        return False
    
    if not test_cascade():
        print("\n❌ Cascade test failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The models are working correctly.")
    print("=" * 60)
    print("\nYou can now use:")
    print("  - python demo.py (for examples)")
    print("  - python demo.py --interactive (for interactive mode)")
    print("  - python text2breaks_inference.py --interactive")
    print("  - python breaks2ssml_inference.py --interactive")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
