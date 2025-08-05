#!/usr/bin/env python3
"""
Inference script for hi-paris/ssml-breaks2ssml-fr-lora
Converts text with symbolic pause markers to proper SSML.

Example:
    Input:  "Bonjour comment allez-vous ?<break/>"
    Output: "<prosody pitch='+0.64%' rate='-1.92%' volume='-10.00%'>Bonjour comment allez-vous ?</prosody><break time='500ms'/>"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from text2breaks_inference import Text2BreaksInference
    from empty_ssml_creation import create_empty_ssml_from_simple_breaks
except ImportError:
    Text2BreaksInference = None
    create_empty_ssml_from_simple_breaks = None

class Breaks2SSMLInference:
    """Inference class for breaks-to-SSML model"""
    
    def __init__(self, model_name="hi-paris/ssml-breaks2ssml-fr-lora", device="auto"):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading base model and tokenizer...")
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
    
    def predict(self, text_with_breaks, max_new_tokens=128, temperature=0.1, do_sample=False):
        """Convert text with simple <break/> tags to proper SSML"""
        if create_empty_ssml_from_simple_breaks is not None:
            empty_ssml_template = create_empty_ssml_from_simple_breaks(text_with_breaks)
            logger.info(f"Empty SSML template: {empty_ssml_template}")
            
            formatted_input = f"### Instruction:\nConvert text Z to text Y.\n\n### Input Z:\n{empty_ssml_template}\n\n### Output Y:\n"
            logger.info(f"Formatted input: {formatted_input}")
        else:
            logger.warning("Empty SSML creation not available, using direct input")
            formatted_input = text_with_breaks
        
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_beams=1,
                early_stopping=False
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if len(full_response) > len(formatted_input):
            ssml_output = full_response[len(formatted_input):].strip()
        else:
            ssml_output = full_response.strip()
        
        if "### Output Y:\n" in formatted_input:
            if "### Output Y:\n" in full_response:
                ssml_output = full_response.split("### Output Y:\n")[-1].strip()
        elif "### SSML:\n" in formatted_input:
            if "### SSML:\n" in full_response:
                ssml_output = full_response.split("### SSML:\n")[-1].strip()
        
        return ssml_output
    
    def predict_batch(self, texts_with_breaks, **kwargs):
        """Convert multiple texts with symbolic breaks to proper SSML"""
        results = []
        for text in texts_with_breaks:
            result = self.predict(text, **kwargs)
            results.append(result)
        return results


class CascadedInference:
    """Combined inference using both models in sequence"""
    
    def __init__(self, 
                 text2breaks_model="hi-paris/ssml-text2breaks-fr-lora",
                 breaks2ssml_model="hi-paris/ssml-breaks2ssml-fr-lora",
                 device="auto"):
        logger.info("Initializing cascaded inference...")
        
        if Text2BreaksInference is None:
            raise ImportError("Text2BreaksInference not available. Please ensure text2breaks_inference.py is accessible.")
        
        self.text2breaks = Text2BreaksInference(text2breaks_model, device)
        self.breaks2ssml = Breaks2SSMLInference(breaks2ssml_model, device)
        
        logger.info("Cascaded inference ready!")
    
    def predict(self, text, **kwargs):
        """Convert plain text to SSML through the cascade"""
        text_with_breaks = self.text2breaks.predict(text, **kwargs)
        logger.info(f"Step 1 - Text with breaks: {text_with_breaks}")
        
        ssml_output = self.breaks2ssml.predict(text_with_breaks, **kwargs)
        logger.info(f"Step 2 - SSML output: {ssml_output}")
        return ssml_output


def main():
    parser = argparse.ArgumentParser(description="Symbolic Breaks to SSML Inference")
    parser.add_argument("text", nargs="?", help="Input text with symbolic breaks")
    parser.add_argument("--model", default="hi-paris/ssml-breaks2ssml-fr-lora", help="HuggingFace model name")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--cascade", action="store_true", help="Use full cascade (text -> breaks -> SSML)")
    
    args = parser.parse_args()
    
    if args.cascade:
        inferencer = CascadedInference(device=args.device)
        input_prompt = "Enter plain French text"
        example_text = "Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour."
    else:
        inferencer = Breaks2SSMLInference(model_name=args.model, device=args.device)
        input_prompt = "Enter text with symbolic breaks (e.g., 'Hello#250 world')"
        example_text = "Bonjour#250 comment vas-tu ?"
    
    if args.interactive:
        print(f"Interactive mode - {input_prompt} (empty line to exit):")
        while True:
            try:
                text = input("\n> ").strip()
                if not text:
                    break
                result = inferencer.predict(text, 
                                          max_new_tokens=args.max_tokens,
                                          temperature=args.temperature)
                print(f"Output: {result}")
            except KeyboardInterrupt:
                break
    else:
        if not args.text:
            print(f"No input text provided. Using example: '{example_text}'")
            args.text = example_text
        
        print(f"Input:  {args.text}")
        result = inferencer.predict(args.text, 
                                  max_new_tokens=args.max_tokens,
                                  temperature=args.temperature)
        print(f"Output: {result}")


if __name__ == "__main__":
    main()
