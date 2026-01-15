"""LLM client adapter with retries and timeouts."""

import httpx
import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI-compatible LLM endpoints."""
    
    def __init__(self, base_url: str, timeout: int = None):
        """Initialize LLM client.
        
        Args:
            base_url: Base URL for the LLM endpoint
            timeout: Timeout in seconds (defaults to LLM_TIMEOUT_S)
        """
        self.base_url = base_url
        self.timeout = timeout or settings.LLM_TIMEOUT_S
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
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
            "model": "default",  # Most OpenAI-compatible APIs accept this
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # Add JSON schema if provided (via system message instruction)
        if json_schema:
            system_msg = {
                "role": "system",
                "content": f"You must respond with valid JSON matching this schema: {json_schema}. Return only the JSON, no additional text."
            }
            messages_with_schema = [system_msg] + messages
            payload["messages"] = messages_with_schema
        
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        
        try:
            logger.debug(f"Calling LLM at {self.base_url} with {len(messages)} messages")
            response = await self.client.post(
                self.base_url,
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

