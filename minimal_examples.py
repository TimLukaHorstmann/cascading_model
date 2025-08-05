#!/usr/bin/env python3
"""
Minimal usage examples for the French SSML cascade models.
These examples can be included in HuggingFace model cards.
"""

# =====================================================================
# Example 1: Text-to-Breaks Model (hi-paris/ssml-text2breaks-fr-lora)
# =====================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter for text-to-breaks
model = PeftModel.from_pretrained(base_model, "hi-paris/ssml-text2breaks-fr-lora")
model.eval()

# Prepare input
instruction = "Convert text to SSML with pauses:"
text = "Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour."
formatted_input = f"### Task:\n{instruction}\n\n### Text:\n{text}\n\n### SSML:\n"

# Generate prediction
inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Extract result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
text_with_breaks = response.split("### SSML:\n")[-1].strip()
print(f"Output: {text_with_breaks}")
# Expected: "Bonjour#250 je m'appelle Bertrand Perier.#500 Je suis avocat à la cour."

# =====================================================================
# Example 2: Breaks-to-SSML Model (hi-paris/ssml-breaks2ssml-fr-lora)
# =====================================================================

# Note: You can reuse the same base_model and tokenizer from above
# or load them fresh as shown below

# Load LoRA adapter for breaks-to-SSML
model = PeftModel.from_pretrained(base_model, "hi-paris/ssml-breaks2ssml-fr-lora")
model.eval()

# Input text with symbolic breaks
input_text = "Bonjour#250 comment vas-tu ?"

# Generate SSML
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Extract result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
ssml_output = response[len(input_text):].strip()
print(f"Output: {ssml_output}")
# Expected: "Bonjour<break time=\"250ms\"/> comment vas-tu ?"

# =====================================================================
# Example 3: Full Cascade (Plain Text → SSML)
# =====================================================================

def full_cascade_example():
    """Complete example showing the full cascade"""
    
    # Load models (reuse tokenizer and base_model)
    text2breaks_model = PeftModel.from_pretrained(base_model, "hi-paris/ssml-text2breaks-fr-lora")
    breaks2ssml_model = PeftModel.from_pretrained(base_model, "hi-paris/ssml-breaks2ssml-fr-lora")
    
    text2breaks_model.eval()
    breaks2ssml_model.eval()
    
    # Input text
    original_text = "Comment allez-vous aujourd'hui ? J'espère que tout va bien."
    print(f"Input: {original_text}")
    
    # Step 1: Text to breaks
    instruction = "Convert text to SSML with pauses:"
    formatted_input = f"### Task:\n{instruction}\n\n### Text:\n{original_text}\n\n### SSML:\n"
    
    inputs = tokenizer(formatted_input, return_tensors="pt").to(text2breaks_model.device)
    with torch.no_grad():
        outputs = text2breaks_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text_with_breaks = response.split("### SSML:\n")[-1].strip()
    print(f"Step 1: {text_with_breaks}")
    
    # Step 2: Breaks to SSML
    inputs = tokenizer(text_with_breaks, return_tensors="pt").to(breaks2ssml_model.device)
    with torch.no_grad():
        outputs = breaks2ssml_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_ssml = response[len(text_with_breaks):].strip()
    print(f"Step 2: {final_ssml}")

# Run the full cascade example
if __name__ == "__main__":
    full_cascade_example()
