# French SSML Cascade Models

This repository contains inference scripts for the French SSML cascade models developed for improving synthetic speech quality via prosody control.

## 🧩 Models

This cascade consists of two specialized LoRA models based on Qwen2.5-7B with a crucial programmatic step:

1. **[hi-paris/ssml-text2breaks-fr-lora](https://huggingface.co/hi-paris/ssml-text2breaks-fr-lora)**: Predicts simple pause markers (e.g., `<break/>`) in French text
2. **Empty SSML Creation (programmatic)**: Converts simple breaks to empty SSML templates with placeholders
3. **[hi-paris/ssml-breaks2ssml-fr-lora](https://huggingface.co/hi-paris/ssml-breaks2ssml-fr-lora)**: Fills SSML templates with proper time attributes and prosody values

### Pipeline Flow

```
Plain Text → Model A → Simple Breaks → Programmatic → Empty SSML → Model B → Full SSML
"Hello"    → "Hello<break/>"        → "<prosody...>Hello</prosody><break time="_ms"/>" → "Hello<break time="250ms"/>"
```

## 🚀 Quick Start

### Installation

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install torch transformers peft accelerate safetensors sentencepiece
```

### Basic Usage

#### 1. Text-to-Breaks (Step 1)
```python
from text2breaks_inference import Text2BreaksInference

# Initialize model
text2breaks = Text2BreaksInference()

# Predict breaks
text = "Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour."
result = text2breaks.predict(text)
print(result)  # "Bonjour#250 je m'appelle Bertrand Perier.#500 Je suis avocat à la cour."
```

#### 2. Breaks-to-SSML (Step 2)
```python
from breaks2ssml_inference import Breaks2SSMLInference

# Initialize model
breaks2ssml = Breaks2SSMLInference()

# Convert to SSML
text_with_breaks = "Bonjour#250 comment vas-tu ?"
result = breaks2ssml.predict(text_with_breaks)
print(result)  # "Bonjour<break time=\"250ms\"/> comment vas-tu ?"
```

#### 3. Full Cascade
```python
from breaks2ssml_inference import CascadedInference

# Initialize cascade
cascade = CascadedInference()

# End-to-end processing
text = "Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour."
result = cascade.predict(text)
print(result)  # Text with proper SSML break tags
```

## 🛠️ Command Line Interface

### Demo Script
Run comprehensive demos with examples:
```bash
python demo.py                    # Run with predefined examples
python demo.py --interactive      # Interactive mode
python demo.py --text "Your text here"
```

### Individual Models
```bash
# Text-to-breaks only
python text2breaks_inference.py "Bonjour comment allez-vous ?"

# Breaks-to-SSML only
python breaks2ssml_inference.py "Bonjour#250 comment allez-vous ?"

# Full cascade
python breaks2ssml_inference.py --cascade "Bonjour comment allez-vous ?"
```

### Interactive Mode
```bash
python text2breaks_inference.py --interactive
python breaks2ssml_inference.py --interactive
python demo.py --interactive
```

## 📋 Examples

### Input/Output Examples

**Input text:**
```
Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour.
```

**Step 1 - Text-to-Breaks:**
```
Bonjour je m'appelle Bertrand Perier.<break/>Je suis avocat à la cour.
```

**Step 2 - Empty SSML Creation (programmatic):**
```
<prosody pitch="_%" rate="_%" volume="_%">Bonjour je m'appelle Bertrand Perier.</prosody><break time="_ms"/><prosody pitch="_%" rate="_%" volume="_%">Je suis avocat à la cour.</prosody>
```

**Step 3 - Breaks-to-SSML:**
```
<prosody pitch="+5%" rate="-7%" volume="-10%">Bonjour je m'appelle Bertrand Perier.</prosody><break time="250ms"/><prosody pitch="+3%" rate="+2%" volume="+10%">Je suis avocat à la cour.</prosody>
```

## ⚙️ Model Details

### Text-to-Breaks Model
- **Base Model**: Qwen/Qwen2.5-7B
- **Method**: LoRA (rank=8, alpha=16)
- **Task**: Predicts symbolic pause markers
- **Input**: Plain French text
- **Output**: Text with symbolic breaks (`#250`, `#500`, etc.)

### Breaks-to-SSML Model
- **Base Model**: Qwen/Qwen2.5-7B
- **Method**: LoRA (rank=8, alpha=16)
- **Task**: Converts symbolic markers to SSML
- **Input**: Text with symbolic breaks
- **Output**: Text with SSML `<break time="..."/>` tags

## 🔧 Configuration

### Generation Parameters

You can customize the generation behavior:

```python
# More deterministic output
result = model.predict(text, temperature=0.1, do_sample=False)

# More creative output
result = model.predict(text, temperature=0.8, do_sample=True)

# Longer outputs
result = model.predict(text, max_new_tokens=512)
```

### Device Selection

```python
# Use specific GPU
model = Text2BreaksInference(device="cuda:0")

# Use CPU
model = Text2BreaksInference(device="cpu")

# Auto-detect (default)
model = Text2BreaksInference(device="auto")
```

## 📊 Performance

### Model Specifications
- **Model Size**: 7B parameters (LoRA adapters only)
- **Languages**: French
- **License**: Apache 2.0

### Evaluation Metrics
- **Pause Insertion Accuracy**: 87.3%
- **RMSE (pause duration)**: 98.5 ms
- **MOS gain (vs. baseline)**: +0.42

## 🎯 Use Cases

- **Text-to-Speech Systems**: Improve prosody in French TTS
- **Speech Synthesis**: Add natural pauses to synthetic speech
- **Accessibility Tools**: Enhance speech output for screen readers
- **Educational Applications**: Create more natural-sounding educational content

## ⚠️ Limitations

- Only supports French text
- Focuses on `<break>` tags (no pitch, rate, or emphasis control yet)
- Optimized primarily for Azure TTS voices
- Requires symbolic pause markers for the second model

## 📖 Citation

If you use these models in your research, please cite:

```bibtex
@inproceedings{ould-ouali2025_improving,
  title     = {Improving Synthetic Speech Quality via SSML Prosody Control},
  author    = {Ould-Ouali, Nassima and Sani, Awais and Bueno, Ruben and Dauvet, Jonah and Horstmann, Tim Luka and Moulines, Eric},
  booktitle = {Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP)},
  year      = {2025},
  publisher = {Springer LNCS}
}
```

## 🤝 Contributing

This repository is maintained by Hi! Paris. For issues or questions about the models, please open an issue on the respective HuggingFace model pages.

## 📜 License

Apache 2.0 License (same as the base Qwen2.5-7B model)
