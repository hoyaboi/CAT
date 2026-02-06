"""
HuggingFace model client for local inference.
"""
from typing import Optional
import os
from .base import LLMClient


class HuggingFaceClient(LLMClient):
    """HuggingFace model client for local inference."""
    
    def __init__(
        self,
        model_id: str,
        hf_access_token: Optional[str] = None,
        use_cuda: bool = True,
        trust_remote_code: bool = True,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        torch_dtype: Optional[str] = None,
    ):
        """
        Initialize HuggingFace client.
        
        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
            hf_access_token: HuggingFace access token (defaults to HUGGINGFACE_TOKEN env var)
            use_cuda: Whether to use CUDA if available
            trust_remote_code: Whether to trust remote code
            max_new_tokens: Maximum new tokens to generate
            device_map: Device mapping strategy
            torch_dtype: Torch data type (e.g., "float16")
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch packages are required. "
                "Install with: pip install transformers torch"
            )
        
        self.model_id = model_id
        self.hf_access_token = hf_access_token or os.getenv("HUGGINGFACE_TOKEN")
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.max_new_tokens = max_new_tokens
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.hf_access_token,
            trust_remote_code=trust_remote_code,
        )
        
        dtype = None
        if torch_dtype == "float16" and self.use_cuda:
            dtype = torch.float16
        elif torch_dtype == "bfloat16" and self.use_cuda:
            dtype = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=self.hf_access_token,
            trust_remote_code=trust_remote_code,
            device_map=device_map if self.use_cuda else None,
            torch_dtype=dtype,
        )
        
        if not self.use_cuda:
            self.model = self.model.to("cpu")
    
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call HuggingFace model."""
        import torch
        
        # Format prompt (adjust based on model's chat template)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        if self.use_cuda:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        max_new_tokens = max_tokens or self.max_new_tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()
        
        return generated_text
