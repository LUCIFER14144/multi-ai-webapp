#!/usr/bin/env python3
"""
Async FastAPI app implementing a 3-AI pipeline:
Researcher -> Writer(s) -> Critic
Includes:
 - async OpenAI calls
 - retries with tenacity
 - optional hook for LangChain/AutoGen orchestration (placeholder)
 - simple frontend to call the API
"""
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import json
from enum import Enum

# Configuration
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "800"))

class AIProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

# Provider configurations
PROVIDER_CONFIGS = {
    AIProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
    },
    AIProvider.GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "default_model": "gemini-1.5-flash",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    },
    AIProvider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-coder"],
        "requires_payment": True  # DeepSeek has no free tier
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-ai")

app = FastAPI(title="Multi-AI Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

class APIKeys(BaseModel):
    openai: Optional[str] = None
    gemini: Optional[str] = None
    deepseek: Optional[str] = None
    
    @validator('*', pre=True)
    def validate_keys(cls, v):
        if v and len(v.strip()) < 10:
            raise ValueError("API key appears to be invalid (too short)")
        return v.strip() if v else None

class TaskRequest(BaseModel):
    prompt: str
    api_keys: APIKeys
    provider: AIProvider = AIProvider.OPENAI
    model: Optional[str] = None
    max_rounds: int = 1
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Prompt must be at least 3 characters long")
        if len(v) > 2000:
            raise ValueError("Prompt too long (max 2000 characters)")
        return v.strip()

class ProviderResult(BaseModel):
    provider: str
    model: str
    answer: str
    error: Optional[str] = None

class TaskResponse(BaseModel):
    final_answer: str
    research_notes: str
    provider_results: List[ProviderResult]  # All providers' answers
    critic_report: str
    winning_provider: str
    winning_model: str

# Retry decorator for transient errors (including rate limits)
def is_rate_limit_error(exc):
    """Check if exception is a rate limit error"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return False

def is_retryable_error(exc):
    """Check if error should be retried"""
    import httpx
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # Retry on rate limits (429) and server errors (5xx)
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False

# Retry with exponential backoff, especially for rate limits
retry_decorator = retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),  # Wait 2s, 4s, 8s, 16s, 30s
    stop=stop_after_attempt(4),  # Try up to 4 times
    reraise=True
)

# Multi-provider AI client
class AIClient:
    def __init__(self, provider: AIProvider, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.config = PROVIDER_CONFIGS[provider]
    
    @retry_decorator
    async def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = MAX_TOKENS) -> str:
        """Unified chat interface for all providers"""
        if self.provider == AIProvider.OPENAI:
            return await self._openai_chat(messages, temperature, max_tokens)
        elif self.provider == AIProvider.GEMINI:
            return await self._gemini_chat(messages, temperature, max_tokens)
        elif self.provider == AIProvider.DEEPSEEK:
            return await self._deepseek_chat(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _openai_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """OpenAI/ChatGPT implementation"""
        url = self.config["base_url"]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    
    async def _gemini_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Google Gemini implementation - uses REST API with URL parameter for key"""
        # Gemini doesn't support system messages, merge with first user message
        gemini_contents = []
        system_prefix = ""
        
        for msg in messages:
            if msg["role"] == "system":
                # Store system message to prepend to first user message
                system_prefix = msg["content"] + "\n\n"
            elif msg["role"] == "user":
                # Prepend system message to first user message only
                content = system_prefix + msg["content"]
                system_prefix = ""  # Clear after first use
                gemini_contents.append({
                    "parts": [{"text": content}]
                })
            elif msg["role"] == "assistant":
                gemini_contents.append({
                    "parts": [{"text": msg["content"]}]
                })
        
        # Gemini API uses URL parameter for authentication
        url = f"{self.config['base_url']}/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        logger.info(f"Gemini API Request - Model: {self.model}, URL: {url[:80]}...")
        logger.debug(f"Gemini payload: {payload}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            # Log response for debugging
            logger.info(f"Gemini Response Status: {response.status_code}")
            
            # Detailed error handling
            if response.status_code == 404:
                error_text = response.text
                logger.error(f"Gemini 404 Error: {error_text}")
                raise Exception(
                    f"❌ Gemini API Key is INVALID or INACTIVE!\n\n"
                    f"Your API key appears to be wrong or not activated.\n\n"
                    f"📌 Get a NEW valid key:\n"
                    f"   1. Visit: https://aistudio.google.com/app/apikey\n"
                    f"   2. Click 'Create API Key'\n"
                    f"   3. Copy the new key (starts with 'AIza...')\n"
                    f"   4. Paste it in the Gemini field above\n\n"
                    f"Note: Keys are free but must be created from Google AI Studio."
                )
            elif response.status_code == 400:
                error_data = response.json()
                logger.error(f"Gemini 400 Error: {error_data}")
                raise Exception(f"Gemini API 400 Error: {error_data.get('error', {}).get('message', 'Bad request')}")
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response text
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                raise Exception(f"Gemini returned no candidates: {data}")
    
    async def _deepseek_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """DeepSeek implementation"""
        url = self.config["base_url"]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

def get_ai_client(provider: AIProvider, api_keys: APIKeys, model: Optional[str] = None) -> AIClient:
    """Factory function to create AI client"""
    # Get API key for the provider
    api_key = None
    if provider == AIProvider.OPENAI:
        api_key = api_keys.openai
    elif provider == AIProvider.GEMINI:
        api_key = api_keys.gemini
    elif provider == AIProvider.DEEPSEEK:
        api_key = api_keys.deepseek
    
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key required for {provider.value}")
    
    # Use provided model or default
    if not model:
        model = PROVIDER_CONFIGS[provider]["default_model"]
    
    # Validate model for provider
    if model not in PROVIDER_CONFIGS[provider]["models"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model} not supported for {provider.value}. Available: {PROVIDER_CONFIGS[provider]['models']}"
        )
    
    return AIClient(provider, api_key, model)

# Improved role prompts (customizable)
RESEARCHER_SYSTEM = (
    "You are Researcher AI. Your job: gather concise, high-quality research notes, "
    "list key facts, assumptions, and 3 reputable sources if available. Flag uncertainties."
)
WRITER_SYSTEM = (
    "You are Writer AI. Your job: convert research notes into clear, engaging content. "
    "Produce a TL;DR (2 lines), suggested headings, and the full polished answer."
)
WRITER_SHORT_SYSTEM = (
    "You are Writer AI (concise). Produce a short, simple explanation (150-220 words) and TL;DR."
)
CRITIC_SYSTEM = (
    "You are Critic AI judging a competition between different AI providers.\n\n"
    "Your job is to:\n"
    "1. Analyze each AI provider's answer for accuracy, clarity, completeness, and consistency with research notes\n"
    "2. Grade each provider's answer (A-F) with brief rationale explaining strengths and weaknesses\n"
    "3. Declare which provider gave the BEST answer and why\n"
    "4. Identify the best elements from each provider\n"
    "5. Create a FINAL MERGED ANSWER that combines the best parts from all providers, fixes any errors, and improves clarity\n\n"
    "Format your response EXACTLY as:\n"
    "=== COMPETITION ANALYSIS ===\n"
    "[Your analysis, grades for each provider, and declaration of winner]\n\n"
    "=== FINAL ANSWER ===\n"
    "[Your improved, merged final answer combining the best from all providers - this is what the user will see]"
)

async def researcher(task: str, ai_client: AIClient):
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nPlease provide research notes, bullets for facts, uncertainties, and 3 sources if available."}
    ]
    return await ai_client.chat(messages, temperature=0.15, max_tokens=600)

async def writer_variant(research_notes: str, ai_client: AIClient, short=False):
    system = WRITER_SHORT_SYSTEM if short else WRITER_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Research notes:\n{research_notes}\n\nPlease produce the requested output."}
    ]
    temp = 0.6 if short else 0.5
    return await ai_client.chat(messages, temperature=temp, max_tokens=700 if not short else 350)

async def critic(research_notes: str, provider_results: List[ProviderResult], ai_client: AIClient):
    """Analyze all provider results and determine the best answer"""
    # Build comparison text
    comparison_text = ""
    for i, result in enumerate(provider_results, 1):
        if result.error:
            comparison_text += f"\n{i}. {result.provider.upper()} ({result.model}): [ERROR - {result.error}]\n"
        else:
            comparison_text += f"\n{i}. {result.provider.upper()} ({result.model}):\n{result.answer}\n\n{'─'*60}\n"
    
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"Research notes:\n{research_notes}\n\n{'═'*60}\nAI PROVIDER COMPETITION RESULTS:\n{'═'*60}\n{comparison_text}\n\nPlease analyze all providers' answers and produce your response in the required format."}
    ]
    return await ai_client.chat(messages, temperature=0.2, max_tokens=1500)

@app.post("/api/generate", response_model=TaskResponse)
async def generate(task: TaskRequest):
    """
    Multi-provider competition: Run all available providers, compare results, and select the best answer.
    """
    try:
        # Determine which providers have API keys
        available_providers = []
        if task.api_keys.openai:
            available_providers.append((AIProvider.OPENAI, task.api_keys.openai))
        if task.api_keys.gemini:
            available_providers.append((AIProvider.GEMINI, task.api_keys.gemini))
        if task.api_keys.deepseek:
            available_providers.append((AIProvider.DEEPSEEK, task.api_keys.deepseek))
        
        if not available_providers:
            raise HTTPException(
                status_code=400,
                detail="No API keys provided. Please provide at least one API key (OpenAI, Gemini, or DeepSeek)."
            )
        
        # Use the first available provider for research (they all see the same research)
        first_provider, first_key = available_providers[0]
        research_client = AIClient(
            first_provider,
            first_key,
            PROVIDER_CONFIGS[first_provider]["default_model"]
        )
        
        # 1) Research phase (shared by all providers)
        logger.info(f"Research phase using {first_provider.value}")
        research_notes = await researcher(task.prompt, research_client)
        
        # 2) Generate answers from ALL available providers concurrently
        logger.info(f"Running competition with {len(available_providers)} providers: {[p.value for p, _ in available_providers]}")
        
        async def get_provider_answer(provider: AIProvider, api_key: str) -> ProviderResult:
            """Get answer from a specific provider"""
            try:
                # Create client for this provider
                model = PROVIDER_CONFIGS[provider]["default_model"]
                client = AIClient(provider, api_key, model)
                
                # Generate answer
                answer = await writer_variant(research_notes, client, short=False)
                
                return ProviderResult(
                    provider=provider.value,
                    model=model,
                    answer=answer,
                    error=None
                )
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {str(e)}")
                return ProviderResult(
                    provider=provider.value,
                    model=PROVIDER_CONFIGS[provider]["default_model"],
                    answer="",
                    error=str(e)[:200]
                )
        
        # Run all providers in parallel
        provider_tasks = [
            get_provider_answer(provider, api_key)
            for provider, api_key in available_providers
        ]
        provider_results = await asyncio.gather(*provider_tasks)
        
        # Filter out results with errors for the critic (but keep them for display)
        successful_results = [r for r in provider_results if not r.error]
        
        if not successful_results:
            raise HTTPException(
                status_code=500,
                detail="All AI providers failed to generate answers. Please check your API keys and try again."
            )
        
        # 3) Use critic to analyze and select the best (use first successful provider for critic)
        logger.info("Critic analyzing all provider results")
        critic_client = research_client  # Reuse the research client for critic
        critic_report = await critic(research_notes, provider_results, critic_client)
        
        # Extract final answer and determine winner from critic report
        final_answer = critic_report
        winning_provider = successful_results[0].provider
        winning_model = successful_results[0].model
        
        if "=== FINAL ANSWER ===" in critic_report:
            parts = critic_report.split("=== FINAL ANSWER ===")
            if len(parts) > 1:
                final_answer = parts[1].strip()
            
            # Try to extract winner from analysis
            analysis_part = parts[0]
            for result in successful_results:
                if result.provider.upper() in analysis_part.upper() and ("winner" in analysis_part.lower() or "best" in analysis_part.lower()):
                    winning_provider = result.provider
                    winning_model = result.model
                    break
        
        return TaskResponse(
            final_answer=final_answer,
            research_notes=research_notes,
            provider_results=provider_results,
            critic_report=critic_report,
            winning_provider=winning_provider,
            winning_model=winning_model
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except httpx.HTTPStatusError as e:
        # Handle specific HTTP status codes from AI providers
        status_code = e.response.status_code
        
        if status_code == 429:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {task.provider.value}. Please wait a moment and try again. "
                       f"If using OpenAI free tier, consider upgrading or trying a different provider (Gemini/DeepSeek)."
            )
        elif status_code == 401:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid API key for {task.provider.value}. Please check your API key and try again."
            )
        elif status_code == 403:
            raise HTTPException(
                status_code=403,
                detail=f"Access forbidden for {task.provider.value}. Your API key may not have the required permissions."
            )
        else:
            raise HTTPException(
                status_code=status_code,
                detail=f"API error from {task.provider.value}: {e.response.text[:200]}"
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The AI provider took too long to respond. Please try again."
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to AI provider. Please check your internet connection and try again."
        )
    except Exception as e:
        logger.exception("Error running pipeline: %s", str(e))
        error_msg = str(e)
        
        # Check if it's a rate limit error mentioned in the message
        if "429" in error_msg or "Too Many Requests" in error_msg:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait a few moments before trying again. "
                       f"Tip: Try using a different AI provider (Gemini or DeepSeek) to avoid rate limits."
            )
        
        raise HTTPException(status_code=500, detail=f"Pipeline error: {error_msg[:300]}")

@app.get("/")
async def root():
    """Serve the frontend interface"""
    return FileResponse("app/frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/providers")
async def get_providers():
    """Get available AI providers and their models"""
    return {
        "providers": {
            provider.value: {
                "name": provider.value.title(),
                "models": config["models"],
                "default_model": config["default_model"],
                "requires_payment": config.get("requires_payment", False),
                "free_tier": not config.get("requires_payment", False)
            }
            for provider, config in PROVIDER_CONFIGS.items()
        }
    }

@app.post("/api/validate-key")
async def validate_api_key(provider: str, api_key: str):
    """Validate an API key for a specific provider"""
    try:
        if provider == "gemini":
            # Test with a simple request
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": "Say 'Hello' in one word"}]
                }]
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{url}?key={api_key}", headers=headers, json=payload)
                if response.status_code == 200:
                    return {"valid": True, "message": "API key is valid"}
                elif response.status_code == 400:
                    error = response.json()
                    return {"valid": False, "message": f"API key invalid: {error.get('error', {}).get('message', 'Unknown error')}"}
                elif response.status_code == 404:
                    return {"valid": False, "message": "API key not found or inactive. Get a new key from https://aistudio.google.com/app/apikey"}
                else:
                    return {"valid": False, "message": f"Unexpected status: {response.status_code}"}
        
        return {"valid": False, "message": "Provider not supported for validation"}
    except Exception as e:
        return {"valid": False, "message": str(e)}

# Optional hooks for LangChain/AutoGen (placeholders)
# If you want to integrate LangChain or AutoGen orchestration, add that logic below.
# Example: call a LangChain LLMChain with prompts for each role instead of direct openai_chat.
