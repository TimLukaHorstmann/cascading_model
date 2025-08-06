#!/usr/bin/env python3
"""
Inference script for hi-paris/ssml-text2breaks-fr-lora
Converts French text to text with symbolic pause markers.

Example:
    Input:  "Bonjour comment allez-vous aujourd'hui ?"
    Output: "Bonjour comment allez-vous aujourd'hui ?<break/>"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Text2BreaksInference:
    """Inference class for text-to-breaks model"""
    
    def __init__(self, model_name="hi-paris/ssml-text2breaks-fr-lora", device="auto"):
        """Initialize the model and tokenizer"""
        logger.info("Loading base model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        logger.info(f"Loading LoRA adapter from {model_name}...")
        self.model = PeftModel.from_pretrained(self.base_model, model_name)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model loaded successfully!")
    
    def predict(self, text, max_new_tokens=256, temperature=0.7, do_sample=True):
        """Convert French text to text with symbolic breaks"""
        if not text or not text.strip():
            return ""
            
        instruction = "Convert text to SSML with pauses:"
        formatted_input = f"### Task:\n{instruction}\n\n### Text:\n{text}\n\n### SSML:\n"
        
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response.split("### SSML:\n")[-1].strip()
        return result
def main():
    parser = argparse.ArgumentParser(description="French Text to Symbolic Breaks Inference")
    parser.add_argument("text", nargs="?", help="Input French text")
    parser.add_argument("--model", default="hi-paris/ssml-text2breaks-fr-lora", 
                       help="HuggingFace model name")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum new tokens")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    inferencer = Text2BreaksInference(model_name=args.model, device=args.device)
    
    if args.interactive:
        print("Interactive mode - Enter French text (empty line to exit):")
        while True:
            try:
                text = input("\n> ").strip()
                if not text:
                    break
                result = inferencer.predict(text, max_new_tokens=args.max_tokens, 
                                          temperature=args.temperature)
                print(f"Output: {result}")
            except KeyboardInterrupt:
                break
    else:
        if not args.text:
            parser.print_help()
            return
        
        print(f"Input:  {args.text}")
        result = inferencer.predict(args.text, max_new_tokens=args.max_tokens, 
                                  temperature=args.temperature)
        print(f"Output: {result}")


if __name__ == "__main__":
    main()
