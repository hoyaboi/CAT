"""
Google Gemini model client.
"""
from typing import Optional
import os
from .base import LLMClient


class GeminiClient(LLMClient):
    """Google Gemini model client."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            model_name: Model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable."
            )
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Gemini API."""
        # Combine system and user prompts
        # Gemini doesn't have separate system/user roles in the same way as OpenAI
        # We'll combine them into a single prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    # Extract text from parts
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    return ''.join(text_parts)
                elif hasattr(candidate, 'content'):
                    return str(candidate.content)
            else:
                return str(response)
        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {e}")

