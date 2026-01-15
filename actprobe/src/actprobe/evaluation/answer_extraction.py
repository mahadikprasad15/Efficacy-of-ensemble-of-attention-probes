"""
Answer extraction logic.
"""

import re
from typing import List
from actprobe.llm.generate import HelperGenerator

class BaseExtractor:
    def extract(self, text: str) -> str:
        raise NotImplementedError

class RegexExtractor(BaseExtractor):
    """
    Heuristic extraction. 
    STRATEGY: 
    1. If "Answer:" matches, take everything after.
    2. Else take the whole text.
    """
    def extract(self, text: str) -> str:
        # Simple split on "Answer:" pattern common in our prompts
        # Our prompts end with "Answer:", so model usually outputs just the answer
        # But sometimes it might repeat "Answer:".
        
        # If the generation starts with "Answer:", strip it.
        pattern = r"^\s*Answer:\s*"
        clean = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        
        # ACT-ViT often just treats the whole generation as the prediction 
        # because the prompt ends with "Answer:". 
        # So we just strip whitespace.
        return clean

class LLMExtractor(BaseExtractor):
    """
    LLM-based extraction.
    Uses a generator to identify the substring.
    """
    def __init__(self, generator: HelperGenerator):
        self.generator = generator
        # Prompt to force extraction
        self.template = (
            "Extract the exact answer (entity, name, or short phrase) from the text below.\n"
            "If the text contains the answer, output ONLY the answer.\n"
            "If the text is irrelevant or refusal, output 'Unanswerable'.\n\n"
            "Text: \"{text}\"\n\n"
            "Extracted Answer:"
        )

    def extract(self, text: str) -> str:
        # Single extraction
        # Note: In production, we'd batch this. For now single loop.
        prompt = self.template.format(text=text)
        # Using a small max_new_tokens for extraction
        output = self.generator.generate([prompt], max_new_tokens=20)[0]
        return output.strip()

    def batch_extract(self, texts: List[str]) -> List[str]:
        prompts = [self.template.format(text=t) for t in texts]
        outputs = self.generator.generate(prompts, max_new_tokens=20)
        return [o.strip() for o in outputs]
