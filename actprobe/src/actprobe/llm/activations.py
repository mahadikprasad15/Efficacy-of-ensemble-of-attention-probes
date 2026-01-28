"""
Activation extraction/running logic.

NOTE ON HIDDEN STATES:
For Llama/decoder-only models, HuggingFace's output_hidden_states returns:
- hidden_states[0]: embeddings
- hidden_states[1:]: post-block hidden states (after RMSNorm → Attn → Residual → RMSNorm → FFN → Residual)
This corresponds to the "post-FFN + residual" activations ACT-ViT uses for probing.
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
    def __init__(self, model_name: str, device: str = None, dtype=torch.float16, hf_token: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        print(f"Loading ActivationRunner: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Set left padding for decoder-only generation
        # Right-padding causes issues with batched generation where padding tokens
        # get mixed into the generated sequence. Left-padding ensures all prompts
        # align at the right edge, and generation appends correctly.
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            token=hf_token
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

        With left-padding (set in __init__), the logic is clean:
        1. Prompts are left-padded to same length
        2. Generation appends to the right
        3. We slice activations to exclude prompt tokens

        Returns:
            (completions, activations_list) where activations are shape (L, T_generated, D)
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # 1. Generate (greedy decoding for determinism, matching ACT-ViT)
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # 2. Decode full sequences (prompt + generation)
        full_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 3. Calculate prompt end indices
        # With left-padding, input_ids.shape[1] is the padded prompt length
        # The actual prompt ends at input_ids.shape[1] in the generated_ids tensor
        prompt_end_indices = []
        clean_completions = []

        for i, prompt in enumerate(prompts):
            # Re-tokenize individual prompt to get its true length (without batch padding)
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
            p_len = prompt_tokens.shape[1]
            prompt_end_indices.append(p_len)

            # Extract completion text (everything after the prompt)
            full_text = full_texts[i]
            if full_text.startswith(prompt):
                completion = full_text[len(prompt):]
            else:
                # Tokenization might add/remove spaces; use full text as fallback
                completion = full_text

            clean_completions.append(completion)

        # 4. Get activations for the full sequence, then slice to generated tokens only
        activations = self.get_activations(full_texts, prompt_end_indices=prompt_end_indices)

        return clean_completions, activations

    @torch.no_grad()
    def get_final_token_activations(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Run forward pass and return hidden states at the FINAL token only.
        
        This is for prompted-probing (Tillman & Mossing 2025 style) where we want 
        the "decision state" at the end of MODEL_INPUT = PASSAGE + DELIM + SUFFIX.
        
        Args:
            texts: List of full model inputs (already contains PASSAGE + DELIM + SUFFIX)
        
        Returns:
            List of tensors shape (L, D) for each input text, where:
            - L = number of layers (excluding embeddings)
            - D = hidden dimension
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Forward pass with output_hidden_states
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # hidden_states is tuple of (L+1) tensors of shape (B, T, D)
        # We take hidden_states[1:] to exclude embeddings
        hs_tuple = outputs.hidden_states[1:]
        stacked = torch.stack(hs_tuple, dim=1)  # (B, L, T, D)
        
        results = []
        for i in range(len(texts)):
            # Get actual length (excluding padding)
            real_len = inputs.attention_mask[i].sum().item()
            
            # Get final token (last non-padded token)
            final_token_idx = int(real_len) - 1
            
            # Extract activation at final token across all layers: (L, D)
            tensor = stacked[i, :, final_token_idx, :]
            results.append(tensor.cpu())
        
        return results

    @torch.no_grad()
    def get_activations_at_position(self, texts: List[str], positions: List[int]) -> List[torch.Tensor]:
        """
        Get hidden states at specified token positions for each text.
        
        Useful for extracting activations at specific points, e.g., 
        at the end of PASSAGE (before DELIM) for delta vector analysis.
        
        Args:
            texts: List of texts
            positions: List of token positions (one per text). Use -1 for final token.
        
        Returns:
            List of tensors shape (L, D) for each input text
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        
        hs_tuple = outputs.hidden_states[1:]
        stacked = torch.stack(hs_tuple, dim=1)  # (B, L, T, D)
        
        results = []
        for i in range(len(texts)):
            real_len = inputs.attention_mask[i].sum().item()
            
            pos = positions[i]
            if pos < 0:
                pos = int(real_len) + pos  # Handle negative indexing
            
            # Clamp to valid range
            pos = max(0, min(pos, int(real_len) - 1))
            
            tensor = stacked[i, :, pos, :]
            results.append(tensor.cpu())
        
        return results
