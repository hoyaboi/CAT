"""
OpenAI GPT model client.
"""
from typing import Optional
import os
from .base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI GPT model client."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            endpoint: API endpoint URL (defaults to OpenAI's default endpoint)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        endpoint = endpoint or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=endpoint)
    
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content
