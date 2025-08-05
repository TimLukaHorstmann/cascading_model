# Quick Start Guide - French SSML Cascade Models

## 📋 Summary

This project provides clean inference scripts for two French SSML models:

1. **hi-paris/ssml-text2breaks-fr-lora**: Text → Symbolic breaks (e.g., `Hello#250 world`)
2. **hi-paris/ssml-breaks2ssml-fr-lora**: Symbolic breaks → SSML (e.g., `Hello<break time="250ms"/> world`)

## 🚀 Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

## 🧪 Test Installation

```bash
python test_models.py
```

## 📖 Usage Examples

### 1. Quick Demo
```bash
python demo.py                    # Run with examples
python demo.py --interactive      # Interactive mode
```

### 2. Individual Models
```bash
# Text to breaks
python text2breaks_inference.py "Bonjour comment allez-vous ?"

# Breaks to SSML  
python breaks2ssml_inference.py "Bonjour#250 comment allez-vous ?"

# Full cascade
python breaks2ssml_inference.py --cascade "Bonjour comment allez-vous ?"
```

### 3. Python API
```python
# Individual models
from text2breaks_inference import Text2BreaksInference
from breaks2ssml_inference import Breaks2SSMLInference

text2breaks = Text2BreaksInference()
breaks2ssml = Breaks2SSMLInference()

# Full cascade
from breaks2ssml_inference import CascadedInference
cascade = CascadedInference()
result = cascade.predict("Bonjour comment allez-vous ?")
```

## 📁 Files Description

| File | Purpose |
|------|---------|
| `text2breaks_inference.py` | Clean inference for text-to-breaks model |
| `breaks2ssml_inference.py` | Clean inference for breaks-to-SSML model |
| `demo.py` | Comprehensive demo script |
| `test_models.py` | Test script to verify installation |
| `minimal_examples.py` | Simple examples for HuggingFace model cards |
| `README.md` | Full documentation |
| `requirements.txt` | Dependencies for pip users |
| `pyproject.toml` | Dependencies for uv users |

## 🎯 Expected Output

**Input:** 
```
Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour.
```

**Step 1 (text2breaks):**
```
Bonjour#250 je m'appelle Bertrand Perier.#500 Je suis avocat à la cour.
```

**Step 2 (breaks2ssml):**
```
Bonjour<break time="250ms"/> je m'appelle Bertrand Perier.<break time="500ms"/> Je suis avocat à la cour.
```

## 🔧 Troubleshooting

1. **Import errors**: Run `python test_models.py` to check dependencies
2. **Model loading fails**: Check internet connection and HuggingFace access
3. **CUDA errors**: Set `device="cpu"` in model initialization
4. **Memory issues**: Reduce `max_new_tokens` or use smaller batch sizes

## 📝 Notes for HuggingFace Model Cards

- Use `minimal_examples.py` for model card code examples
- Both models are LoRA adapters requiring the base Qwen/Qwen2.5-7B model
- Models are optimized for French text
- Default temperature: 0.7 for text2breaks, 0.3 for breaks2ssml
