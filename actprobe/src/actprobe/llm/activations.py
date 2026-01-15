"""
Activation extraction/running logic.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

class ActivationRunner:
    """
    Runs forward passes to extract hidden states.
    Optimized for batch processing.
    """
    def __init__(self, model_name: str, device: str = None, dtype=torch.float16):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Load model for access to hidden states
        # Pipeline is great for generation but for activations we want direct access usually.
        # But we can also use model directly.
        
        print(f"Loading ActivationRunner: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def get_activations(self, texts: List[str], prompt_end_indices: Optional[List[int]] = None) -> List[torch.Tensor]:
        """
        Run forward pass and return hidden states stack (L, T, D).
        If prompt_end_indices is provided, slices to ONLY generated tokens specific to ACT-ViT.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Forward pass with output_hidden_states
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # hidden_states is tuple of (L+1) tensors of shape (B, T, D)
        # HF hidden_states[0] is embeddings. hidden_states[-1] is last layer.
        # We take hidden_states[1:] to exclude embeddings.
        hs_tuple = outputs.hidden_states[1:] 
        stacked = torch.stack(hs_tuple, dim=1) # (B, L, T, D)
        
        results = []
        for i in range(len(texts)):
            # Determine actual length (excluding padding)
            real_len = inputs.attention_mask[i].sum().item()
            
            # Determine start index (prompt end)
            start_idx = 0
            if prompt_end_indices and len(prompt_end_indices) > i:
                 start_idx = prompt_end_indices[i]
            
            # Slice: Layers, Start:End, Dim
            tensor = stacked[i, :, start_idx:real_len, :] 
            results.append(tensor.cpu()) # Move to CPU
            
        return results

    @torch.no_grad()
    def generate_and_get_activations(self, prompts: List[str], max_new_tokens: int = 128) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate completions, then run forward pass to get activations for the GENERATED part only.
        Returns:
            (completions, activations_list)
        """
        input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device).input_ids
        
        # 1. Generate
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, # ACT-ViT typically uses greedy for probing
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 2. Identify prompt lengths to slice later
        # generated_ids contains [Prompt + Generated] usually (decoder-only)
        # We need to know where prompt ends.
        # Since we batched, prompts might be padded. 
        # But 'generated_ids' usually handles this by appending.
        # Reliable way: input_ids shape is (B, L_in). generate returns (B, L_out).
        # But padding makes L_in variable in reality if we didn't left-pad?
        # Standard generate usually appends. 
        # Let's assume left-padding for generation if batching.
        
        # Actually, tokenizer(padding=True) defaults to right padding. 
        # Creating issues for generation if not careful.
        # But for 'generate', we usually need LEFT padding.
        # NOTE: AutoTokenizer usually sets padding_side='right'. 
        # We should check and set to 'left' for generation?
        
        # Re-tokenization for safety? Or just handle slicing carefully.
        # Simple approach: decode generated_ids, strip prompt text?
        # NO, we want activations aligned to tokens.
        
        # Input: [P1, P2, P3] (padded right)
        # Generate -> [P1..G1, P2..G2...]
        # The prompt part in generated_ids refers to the input tokens.
        
        full_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions = []
        
        # Calculate indices
        prompt_lengths = []
        input_len = input_ids.shape[1] 
        
        # Use simpler approach:
        # Prompt Length = input_ids.shape[1] (if right padded, this includes pads!)
        # We must use attention_mask of inputs to find real prompt end?
        # Actually, if we use left padding, prompt ends at input_len.
        
        # Let's force left padding for this method locally?
        # Or just compute offsets based on matched text.
        
        # Robust Logic:
        # 1. Decode full text.
        # 2. Find prompt string len? No, token len.
        
        # Let's rely on the fact that generate returns raw IDs.
        # generated_ids[:, :input_len] are the prompt tokens (if no left padding shift happened).
        # Actually, if we right-padded inputs, generate might append after pads? No, it usually handles attention mask.
        
        # For simplicity in this script, let's assume batch_size=1 or handle padding carefully.
        # OR: Just run a forward pass on the `full_texts` and re-compute prompt length via tokenizer.
        
        prompt_end_indices = []
        clean_completions = []
        
        for i, text in enumerate(full_texts):
             # Re-tokenize prompt to find length?
             # Or just use the input prompt string.
             prompt_tokens = self.tokenizer(prompts[i], return_tensors="pt").input_ids
             p_len = prompt_tokens.shape[1]
             prompt_end_indices.append(p_len)
             
             # Completion text
             # Approximate: split by prompt?
             # text[len(prompts[i]):] ?
             # Be careful with spacing.
             
             # Let's define completion as text - prompt.
             if text.startswith(prompts[i]):
                 comp = text[len(prompts[i]):]
             else:
                 # Fallback
                 comp = text
                 
             clean_completions.append(comp)

        # 3. Get Activations
        activations = self.get_activations(full_texts, prompt_end_indices=prompt_end_indices)
        
        return clean_completions, activations
