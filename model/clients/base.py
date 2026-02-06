"""
Base interface for LLM clients.
"""
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Base interface for LLM clients."""
    
    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call the LLM with system and user prompts.
        
        Args:
            system_prompt: System prompt string
            user_prompt: User prompt string
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (None for model default)
            
        Returns:
            Model response string
        """
        pass
