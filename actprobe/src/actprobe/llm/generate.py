"""
Generation logic for ACT-ViT pipeline.
Supports Local HuggingFace models and Cerebras API (placeholder).
"""

import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class HelperGenerator:
    """Base class for Generators."""
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError

class LocalGenerator(HelperGenerator):
    """Generator using local HuggingFace transformers."""
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading local model: {model_name} on {self.device}")
        try:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

    def generate(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        # Batching handled by pipeline usually, but simplistic for now
        outputs = self.pipe(
            prompts, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, # Greedier generation often better for factual QA
            return_full_text=False,
            **kwargs
        )
        return [out[0]['generated_text'] for out in outputs]

class CerebrasGenerator(HelperGenerator):
    """Generator using Cerebras API."""
    def __init__(self, model_name: str = "llama3.1-70b", api_key: str = None):
        # NOTE: Requires cerebras package
        try:
            from cerebras.cloud.sdk import Cerebras
            import os
            self.client = Cerebras(api_key=api_key or os.environ.get("CEREBRAS_API_KEY"))
            self.model_name = model_name
        except ImportError:
            raise ImportError("cerebras.cloud.sdk not installed.")
        except Exception as e:
            raise ValueError(f"Failed to init Cerebras client: {e}")

    def generate(self, prompts: List[str], max_tokens: int = 128, **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            # Cerebras SDK usage (approximated, verification needed for exact syntax usually openai-like)
            # Assuming openai-compatible interface as is common
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                    **kwargs
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"Cerebras generation failed: {e}")
                results.append("")
        return results
