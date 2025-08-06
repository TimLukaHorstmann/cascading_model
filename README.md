# French SSML Cascade Models

This repository provides inference scripts for French SSML cascade models that improve synthetic speech quality through intelligent prosody control.

## 🧩 Architecture

The cascade consists of two specialized LoRA models based on Qwen2.5-7B with a programmatic processing step:

1. **[hi-paris/ssml-text2breaks-fr-lora](https://huggingface.co/hi-paris/ssml-text2breaks-fr-lora)**: Predicts natural pause locations in French text
2. **Empty SSML Creation (programmatic)**: Converts simple breaks to structured SSML templates
3. **[hi-paris/ssml-breaks2ssml-fr-lora](https://huggingface.co/hi-paris/ssml-breaks2ssml-fr-lora)**: Fills templates with appropriate prosody values

### Pipeline Flow

```
Plain Text → Text2Breaks → Simple Breaks → Programmatic → Empty SSML → Breaks2SSML → Full SSML
"Bonjour"  → "Bonjour<break/>"           → "<prosody...>Bonjour</prosody><break time="_ms"/>" → "Bonjour<break time="250ms"/>"
```

## 🚀 Quick Start

### Installation

Using uv (recommended):
```bash
git clone <repository-url>
cd cascading_model
uv sync
```

Using pip:
```bash
git clone <repository-url>
cd cascading_model
pip install -r requirements.txt
```

### Basic Usage

### Python API

#### Full Cascade (Recommended)
```python
from breaks2ssml_inference import CascadedInference

# Initialize cascade
cascade = CascadedInference()

# Convert plain text to SSML
text = "Bonjour comment allez-vous aujourd'hui ?"
result = cascade.predict(text)
print(result)  # Full SSML with prosody and break tags
```

#### Individual Steps
```python
# Step 1: Text to breaks
from text2breaks_inference import Text2BreaksInference
text2breaks = Text2BreaksInference()
breaks_result = text2breaks.predict("Bonjour comment allez-vous aujourd'hui ?")

# Step 2: Breaks to SSML  
from breaks2ssml_inference import Breaks2SSMLInference
breaks2ssml = Breaks2SSMLInference()
ssml_result = breaks2ssml.predict(breaks_result)
```

## 🛠️ Usage

### Quick Start
```bash
# Run examples
python demo.py

# Interactive mode
python demo.py --interactive

# Test installation
python test_models.py
```

### Command Line Interface

```bash
# Full cascade
python breaks2ssml_inference.py --cascade "Bonjour comment allez-vous ?"

# Individual models
python text2breaks_inference.py "Bonjour comment allez-vous ?"
python breaks2ssml_inference.py "Bonjour comment allez-vous ?<break/>"

# Interactive modes
python text2breaks_inference.py --interactive
python breaks2ssml_inference.py --interactive --cascade
```

## 📋 Examples

### Input/Output Examples

**Input text:**
```
Bonjour comment allez-vous aujourd'hui ?
```

**Step 1 - Text-to-Breaks:**
```
Bonjour comment allez-vous aujourd'hui ?<break/>
```

**Step 2 - Empty SSML Creation (programmatic):**
```
<prosody pitch="_%" rate="_%" volume="_%">Bonjour comment allez-vous aujourd'hui ?</prosody><break time="_ms"/>
```

**Step 3 - Breaks-to-SSML:**
```
<prosody pitch="+2.5%" rate="-1.2%" volume="-5.0%">Bonjour comment allez-vous aujourd'hui ?</prosody><break time="300ms"/>
```

## ⚙️ Configuration

### Generation Parameters
```python
# More deterministic output
result = model.predict(text, temperature=0.1, do_sample=False)

# More varied output  
result = model.predict(text, temperature=0.8, do_sample=True)

# Control output length
result = model.predict(text, max_new_tokens=256)
```

### Device Selection
```python
# Specific GPU
model = Text2BreaksInference(device="cuda:0")

# CPU only
model = Text2BreaksInference(device="cpu")

# Auto-detect (default)
model = Text2BreaksInference(device="auto")
```

## 📊 Model Information

- **Base Model**: Qwen/Qwen2.5-7B with LoRA adapters
- **Language**: French
- **License**: Apache 2.0
- **Use Cases**: TTS prosody control, speech synthesis, accessibility tools

## 📖 Citation

If you use these models in your research, please cite:

```bibtex
@inproceedings{ould-ouali2025_improving,
  title     = {Improving Synthetic Speech Quality via SSML Prosody Control},
  author    = {Ould-Ouali, Nassima and Sani, Awais and Bueno, Ruben and Dauvet, Jonah and Horstmann, Tim Luka and Moulines, Eric},
  booktitle = {Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP)},
  year      = {2025},
  url       = {https://huggingface.co/hi-paris}
}
```

## 🤝 Contributing

This repository is maintained by Hi! Paris. For issues or questions about the models, please open an issue on the respective HuggingFace model pages.

## 📜 License

Apache 2.0 License (same as the base Qwen2.5-7B model)
