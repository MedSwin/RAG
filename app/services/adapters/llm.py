"""LLM client adapter with retries and timeouts."""

import httpx
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.adapters.rate_limit import request_with_model_rate_limit
from app.services.prompts.structured import schema_instruction

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI-compatible LLM endpoints."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize LLM client.
        
        Args:
            base_url: Base URL for the LLM endpoint
            timeout: Timeout in seconds (defaults to LLM_TIMEOUT_S)
        """
        self.base_url = base_url
        self.timeout = timeout or settings.LLM_TIMEOUT_S
        self.model = model or (settings.CLOUD_MODEL if settings.CLOUD_MODE else "default")
        self.api_key = api_key or (settings.AZURE_AI_FOUNDRY_API_KEY if settings.CLOUD_MODE else None)
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.rate_limit_key = f"llm:{self.base_url}:{self.model}"
    
    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call LLM endpoint with retries and error handling.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            json_schema: Optional JSON schema for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            request_id: Optional request ID for tracing
            
        Returns:
            Response dict with 'content' and optional 'token_count'
            
        Raises:
            httpx.HTTPError: If request fails after retries
        """
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if not self.model.lower().startswith("gpt-5"):
            payload["temperature"] = temperature
        
        if max_tokens:
            # Root Cause vs Logic: Azure OpenAI-compatible GPT-5.x deployments
            # reject the legacy max_tokens field. Use the model-family specific
            # completion budget while preserving local/legacy payloads.
            if self.model.lower().startswith("gpt-5"):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        
        # Add JSON schema if provided (via system message instruction)
        if json_schema:
            system_msg = {
                "role": "system",
                "content": schema_instruction(json_schema)
            }
            messages_with_schema = [system_msg] + messages
            payload["messages"] = messages_with_schema
        
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        if request_id:
            headers["X-Request-ID"] = request_id
        
        try:
            logger.debug(f"Calling LLM at {self.base_url} with {len(messages)} messages")
            response = await request_with_model_rate_limit(
                self.client,
                self.base_url,
                rate_limit_key=self.rate_limit_key,
                logger=logger,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract content from OpenAI-compatible response
            content = ""
            token_count = None
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "text" in choice:
                    content = choice["text"]
            
            if "usage" in data:
                token_count = data["usage"].get("total_tokens")
            
            return {
                "content": content,
                "token_count": token_count,
                "raw_response": data
            }
            
        except httpx.HTTPError as e:
            logger.error(f"LLM request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
