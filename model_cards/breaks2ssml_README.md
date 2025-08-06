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

# 🗣️ French Breaks-to-SSML LoRA Model

**hi-paris/ssml-breaks2ssml-fr-lora** is a LoRA adapter fine-tuned on Qwen2.5-7B to convert text with symbolic `<break/>` markers into rich SSML markup with prosody control (pitch, rate, volume) and precise break timing.

This is the **second stage** of a two-step SSML cascade pipeline for improving French text-to-speech prosody control.

> 📄 **Paper**: *"Improving Synthetic Speech Quality via SSML Prosody Control"*  
> **Authors**: Nassima Ould-Ouali, Awais Sani, Ruben Bueno, Jonah Dauvet, Tim Luka Horstmann, Eric Moulines  
> **Conference**: ICNLSP 2025  
> 🔗 **Demo & Audio Samples**: https://horstmann.tech/ssml-prosody-control/

## 🧩 Pipeline Overview

| Stage | Model | Purpose |
|-------|-------|---------|
| 1️⃣ | [hi-paris/ssml-text2breaks-fr-lora](https://huggingface.co/hi-paris/ssml-text2breaks-fr-lora) | Predicts natural pause locations |
| 2️⃣ | **hi-paris/ssml-breaks2ssml-fr-lora** | Converts breaks to full SSML with prosody |

## ✨ Example

**Input:**
```
Bonjour comment allez-vous ?<break/>
```

**Output:**
```
<prosody pitch="+2.5%" rate="-1.2%" volume="-5.0%">Bonjour comment allez-vous ?</prosody><break time="300ms"/>
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
model = PeftModel.from_pretrained(base_model, "hi-paris/ssml-breaks2ssml-fr-lora")

# Prepare input (text with <break/> markers)
text_with_breaks = "Bonjour comment allez-vous ?<break/>"
formatted_input = f"### Task:\nConvert text to SSML with pauses:\n\n### Text:\n{text_with_breaks}\n\n### SSML:\n"

# Generate
inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
result = response.split("### SSML:\n")[-1].strip()
print(result)
```

### Production Usage (Recommended)

For production use with memory optimization, see our [inference repository](https://github.com/TimLukaHorstmann/cascading_model):

```python
from breaks2ssml_inference import Breaks2SSMLInference

# Memory-efficient shared model approach
model = Breaks2SSMLInference()
result = model.predict("Bonjour comment allez-vous ?<break/>")
```

## 🔧 Full Cascade Example

```python
from breaks2ssml_inference import CascadedInference

# Initialize full pipeline (memory efficient - single base model)
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

| Metric | Score |
|--------|-------|
| Pause Insertion Accuracy | 87.3% |
| RMSE (pause duration) | 98.5 ms |
| MOS gain (vs. baseline) | +0.42 |

*Evaluation performed on held-out French validation set with annotated SSML pauses. Mean Opinion Score (MOS) improvements assessed using TTS outputs with Azure Henri voice, rated by 30 native French speakers.*

## 🎯 SSML Features Generated

- **Prosody Control**: Dynamic pitch, rate, and volume adjustments
- **Break Timing**: Precise pause durations (e.g., `<break time="300ms"/>`)
- **Contextual Adaptation**: Prosody values adapted to semantic content

## ⚠️ Limitations

- Optimized primarily for Azure TTS voices (e.g., `fr-FR-HenriNeural`)
- Requires input text with `<break/>` markers (use Stage 1 model for automatic prediction)
- Currently supports break tags only (pitch/rate/volume via prosody wrapper)

## 🔗 Resources

- **Full Pipeline Code**: https://github.com/TimLukaHorstmann/cascading_model
- **Interactive Demo**: [Colab Notebook](https://colab.research.google.com/drive/1bFcbJQY9OuY0_zlscqkf9PIgd3dUrIKs?usp=sharing)
- **Stage 1 Model**: [hi-paris/ssml-text2breaks-fr-lora](https://huggingface.co/hi-paris/ssml-text2breaks-fr-lora)

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
