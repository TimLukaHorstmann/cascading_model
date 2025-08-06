---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B
library_name: peft
language:
- fr
tags:
- lora
- peft
- ssml
- text-to-speech
- qwen2.5
---

# 🗣️ French Text-to-Breaks LoRA Model

**hi-paris/ssml-text2breaks-fr-lora** is a LoRA adapter fine-tuned on Qwen2.5-7B to predict natural pause locations in French text by adding symbolic `<break/>` markers.

This is the **first stage** of a two-step SSML cascade pipeline for improving French text-to-speech prosody control.

> 📄 **Paper**: *"Improving Synthetic Speech Quality via SSML Prosody Control"*  
> **Authors**: Nassima Ould-Ouali, Awais Sani, Ruben Bueno, Jonah Dauvet, Tim Luka Horstmann, Eric Moulines  
> **Conference**: ICNLSP 2025  
> 🔗 **Demo & Audio Samples**: https://horstmann.tech/ssml-prosody-control/

## 🧩 Pipeline Overview

| Stage | Model | Purpose |
|-------|-------|---------|
| 1️⃣ | **hi-paris/ssml-text2breaks-fr-lora** | Predicts natural pause locations |
| 2️⃣ | [hi-paris/ssml-breaks2ssml-fr-lora](https://huggingface.co/hi-paris/ssml-breaks2ssml-fr-lora) | Converts breaks to full SSML with prosody |

## ✨ Example

**Input:**
```
Bonjour comment allez-vous aujourd'hui ?
```

**Output:**
```
Bonjour comment allez-vous aujourd'hui ?<break/>
```

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers peft accelerate
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "hi-paris/ssml-text2breaks-fr-lora")

# Prepare input
text = "Bonjour comment allez-vous aujourd'hui ?"
formatted_input = f"### Task:\nConvert text to SSML with pauses:\n\n### Text:\n{text}\n\n### SSML:\n"

# Generate
inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
result = response.split("### SSML:\n")[-1].strip()
print(result)  # "Bonjour comment allez-vous aujourd'hui ?<break/>"
```

### Production Usage (Recommended)

For production use with memory optimization and full cascade, see our [inference repository](https://github.com/TimLukaHorstmann/cascading_model):

```python
from text2breaks_inference import Text2BreaksInference

# Memory-efficient shared model approach
model = Text2BreaksInference()
result = model.predict("Bonjour comment allez-vous aujourd'hui ?")
```

## 🔧 Full Cascade Example

```python
from breaks2ssml_inference import CascadedInference

# Initialize full pipeline (memory efficient)
cascade = CascadedInference()

# Convert plain text directly to full SSML
text = "Bonjour comment allez-vous aujourd'hui ?"
ssml_output = cascade.predict(text)
print(ssml_output)  
# Output: '<prosody pitch="+2.5%" rate="-1.2%" volume="-5.0%">Bonjour comment allez-vous aujourd'hui ?</prosody><break time="300ms"/>'
```

## 🧠 Model Details

- **Base Model**: [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 8, Alpha: 16
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Training**: 5 epochs, batch size 1 with gradient accumulation
- **Language**: French
- **Model Size**: 7B parameters (LoRA adapter: ~81MB)
- **License**: Apache 2.0

## 📊 Performance

The model achieves high accuracy in predicting natural pause locations in French text, contributing to improved prosody in text-to-speech synthesis when combined with the second-stage model.

## 🔗 Resources

- **Full Pipeline Code**: https://github.com/TimLukaHorstmann/cascading_model
- **Interactive Demo**: [Colab Notebook](https://colab.research.google.com/drive/1bFcbJQY9OuY0_zlscqkf9PIgd3dUrIKs?usp=sharing)
- **Stage 2 Model**: [hi-paris/ssml-breaks2ssml-fr-lora](https://huggingface.co/hi-paris/ssml-breaks2ssml-fr-lora)

## 📖 Citation

```bibtex
@inproceedings{ould-ouali2025_improving,
  title     = {Improving Synthetic Speech Quality via SSML Prosody Control},
  author    = {Ould-Ouali, Nassima and Sani, Awais and Bueno, Ruben and Dauvet, Jonah and Horstmann, Tim Luka and Moulines, Eric},
  booktitle = {Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP)},
  year      = {2025},
  url       = {https://huggingface.co/hi-paris}
}
```

## 📜 License

Apache 2.0 License (same as the base Qwen2.5-7B model)
