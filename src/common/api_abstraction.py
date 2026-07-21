"""
API Abstraction Layer for Philosopher Agents
Allows switching between Google Gemini and LLaMA (Ollama) backends
"""

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from config.settings import ACTIVE_API, LLM_MODEL, TEMPERATURE, MAX_TOKENS


class PhilosopherAPI:
    """Base class for LLM API abstraction"""
    
    def get_llm(self, model_override: str = None) -> BaseChatModel | BaseLLM:
        """Return configured LLM instance
        
        Args:
            model_override: Optional model name to override default
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def get_model_name(self) -> str:
        """Return model name for logging"""
        raise NotImplementedError("This method should be overridden by subclasses")


class GoogleAPI(PhilosopherAPI):
    """Google Gemini API implementation"""
    
    def __init__(self, model: str = LLM_MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def get_llm(self, model_override: str = None) -> ChatGoogleGenerativeAI:
        """Return configured Google Gemini LLM
        
        Args:
            model_override: Optional model name to override self.model
        """
        model_name = model_override or self.model
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
    
    def get_model_name(self) -> str:
        return f"Google Gemini ({self.model})"


class LlamaAPI(PhilosopherAPI):
    """LLaMA via Ollama implementation"""
    
    def __init__(self, model: str = "llama3.2", temperature: float = TEMPERATURE):
        self.model = model
        self.temperature = temperature
    
    def get_llm(self, model_override: str = None) -> Ollama:
        """Return configured Ollama LLaMA LLM
        
        Args:
            model_override: Optional model name to override self.model
        """
        model_name = model_override or self.model
        return Ollama(
            model=model_name,
            temperature=self.temperature
        )
    
    def get_model_name(self) -> str:
        return f"LLaMA via Ollama ({self.model})"


def get_philosopher_api() -> PhilosopherAPI:
    """
    Factory function to get configured API based on ACTIVE_API setting.
    
    Returns:
        PhilosopherAPI: Configured API instance (GoogleAPI or LlamaAPI)
    
    Raises:
        ValueError: If ACTIVE_API is not 'google' or 'llama'
    """
    if ACTIVE_API == "google":
        return GoogleAPI()
    elif ACTIVE_API == "llama":
        return LlamaAPI()
    else:
        raise ValueError(
            f"Unsupported API: {ACTIVE_API}. "
            f"Set ACTIVE_API to 'google' or 'llama' in config/settings.py or environment variable."
        )