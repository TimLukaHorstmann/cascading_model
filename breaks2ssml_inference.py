#!/usr/bin/env python3
"""
Clean inference script for hi-paris/ssml-breaks2ssml-fr-lora
Converts text with symbolic pause markers to proper SSML with <break time="..."/> tags.

Usage:
    python breaks2ssml_inference.py "Bonjour#250 comment vas-tu ?"
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import for cascaded inference
try:
    from text2breaks_inference import Text2BreaksInference
    from empty_ssml_creation import create_empty_ssml_from_simple_breaks, create_empty_ssml_multiline
except ImportError:
    # Handle case where dependencies might not be available
    Text2BreaksInference = None
    create_empty_ssml_from_simple_breaks = None
    create_empty_ssml_multiline = None

class Breaks2SSMLInference:
    """Inference class for breaks-to-SSML model"""
    
    def __init__(self, model_name="hi-paris/ssml-breaks2ssml-fr-lora", device="auto"):
        """
        Initialize the model and tokenizer
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading base model and tokenizer...")
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        logger.info(f"Loading LoRA adapter from {model_name}...")
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, model_name)
        self.model.eval()
        
        # Add pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model loaded successfully!")
    
    def predict(self, text_with_breaks, max_new_tokens=128, temperature=0.1, do_sample=False):
        """
        Convert text with simple <break/> tags to proper SSML
        
        Args:
            text_with_breaks: Input text with simple break tags (e.g., "Bonjour<break/>comment vas-tu ?")
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy, lower for more deterministic)
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Text with SSML break tags (e.g., "Bonjour<break time=\"250ms\"/>comment vas-tu ?")
        """
        # CRUCIAL: Apply Empty SSML Creation step first
        # Convert simple breaks to empty SSML template as per the cascade diagram
        if create_empty_ssml_from_simple_breaks is not None:
            empty_ssml_template = create_empty_ssml_from_simple_breaks(text_with_breaks)
            logger.info(f"Empty SSML template: {empty_ssml_template}")
            formatted_input = empty_ssml_template
        else:
            # Fallback to direct input if empty_ssml_creation is not available
            logger.warning("Empty SSML creation not available, using direct input")
            formatted_input = text_with_breaks
        
        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        
        # Generate prediction with more conservative settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_beams=1,  # Greedy decoding
                early_stopping=True
            )
        
        # Decode and extract the generated part
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the input)
        if len(full_response) > len(formatted_input):
            ssml_output = full_response[len(formatted_input):].strip()
        else:
            ssml_output = full_response.strip()
        
        # Clean up the output - try to extract just the first line if there's excessive generation
        lines = ssml_output.split('\n')
        if len(lines) > 1 and lines[0].strip():
            # Take the first non-empty line that looks like our expected output
            ssml_output = lines[0].strip()
            
        return ssml_output
    
    def predict_batch(self, texts_with_breaks, **kwargs):
        """
        Convert multiple texts with symbolic breaks to proper SSML
        
        Args:
            texts_with_breaks: List of input texts with symbolic breaks
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of texts with SSML break tags
        """
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
        """
        Initialize both models for cascaded inference
        
        Args:
            text2breaks_model: Text-to-breaks model name
            breaks2ssml_model: Breaks-to-SSML model name
            device: Device to use
        """
        logger.info("Initializing cascaded inference...")
        
        # Check if Text2BreaksInference is available
        if Text2BreaksInference is None:
            raise ImportError("Text2BreaksInference not available. Please ensure text2breaks_inference.py is accessible.")
        
        # Initialize both models
        self.text2breaks = Text2BreaksInference(text2breaks_model, device)
        self.breaks2ssml = Breaks2SSMLInference(breaks2ssml_model, device)
        
        logger.info("Cascaded inference ready!")
    
    def predict(self, text, **kwargs):
        """
        Convert plain text to SSML through the cascade
        
        Args:
            text: Plain French text
            **kwargs: Additional arguments for generation
            
        Returns:
            SSML with break tags
        """
        # Step 1: Add symbolic breaks
        text_with_breaks = self.text2breaks.predict(text, **kwargs)
        logger.info(f"Step 1 - Text with breaks: {text_with_breaks}")
        
        # Step 2: Convert to SSML
        ssml_output = self.breaks2ssml.predict(text_with_breaks, **kwargs)
        logger.info(f"Step 2 - SSML output: {ssml_output}")
        
        return ssml_output


def main():
    parser = argparse.ArgumentParser(description="Symbolic Breaks to SSML Inference")
    parser.add_argument("text", nargs="?", help="Input text with symbolic breaks (e.g., 'Hello#250 world')")
    parser.add_argument("--model", default="hi-paris/ssml-breaks2ssml-fr-lora", 
                       help="HuggingFace model name")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--cascade", action="store_true", help="Use full cascade (text -> breaks -> SSML)")
    
    args = parser.parse_args()
    
    if args.cascade:
        # Use full cascade
        inferencer = CascadedInference(device=args.device)
        input_prompt = "Enter plain French text"
        example_text = "Bonjour je m'appelle Bertrand Perier. Je suis avocat à la cour."
    else:
        # Use only breaks2ssml model
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
            # Example if no text provided
            print(f"No input text provided. Using example: '{example_text}'")
            args.text = example_text
        
        print(f"Input:  {args.text}")
        result = inferencer.predict(args.text, 
                                  max_new_tokens=args.max_tokens,
                                  temperature=args.temperature)
        print(f"Output: {result}")


if __name__ == "__main__":
    main()
