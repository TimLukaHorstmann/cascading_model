#!/usr/bin/env python3
"""
Shared model management for French SSML cascade models.
Efficiently manages the base Qwen2.5-7B model with interchangeable LoRA adapters.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)

class SharedModelManager:
    def __init__(self, device="auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        self.base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True, # Might be necessary for environments with limited GPU memory, e.g. colab
            max_memory={0: "15GiB", "cpu": "30GiB"}, # Might be necessary for environments with limited GPU memory, e.g. colab
            # offload_folder="./offload" # Might be necessary for environments with limited GPU memory, e.g. colab
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.peft_model = None
        self._loaded = set()  # adapter aliases we’ve loaded

    def use_adapter(self, repo_or_path: str, alias: str):
        """
        Ensure `alias` is available in the single wrapper and activate it.
        """
        if self.peft_model is None:
            logger.info(f"Loading first adapter {alias} from {repo_or_path}")
            self.peft_model = PeftModel.from_pretrained(
                self.base,
                repo_or_path,
                adapter_name=alias,
                is_trainable=False,
                # offload_folder="./offload" # Might be necessary for environments with limited GPU memory, e.g. colab
            )
            self._loaded.add(alias)
        elif alias not in self._loaded:
            logger.info(f"Loading additional adapter {alias} from {repo_or_path}")
            self.peft_model.load_adapter(
                repo_or_path,
                adapter_name=alias,
                is_trainable=False,
                # offload_folder="./offload" # Might be necessary for environments with limited GPU memory, e.g. colab
            )
            self._loaded.add(alias)

        logger.info(f"Activating adapter {alias}")
        self.peft_model.set_adapter(alias)   # disables others
        self.peft_model.eval()
        return self.peft_model, self.tokenizer

# Global shared instance
_shared_manager = None

def get_shared_manager(device="auto"):
    """Get or create the global shared model manager"""
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = SharedModelManager(device=device)
    return _shared_manager
