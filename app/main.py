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
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import json
from enum import Enum
import jwt
import hashlib
import uuid

# Configuration
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "800"))
JWT_SECRET = os.environ.get("JWT_SECRET", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Simple in-memory storage (replace with database in production)
users_db = {}
user_history_db = {}

# Auth models
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match("^[a-zA-Z0-9_]{3,20}$", v):
            raise ValueError('Username must be 3-20 characters, alphanumeric and underscore only')
        return v
    
    @validator('email')
    def email_valid(cls, v):
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError('Invalid email format')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: str
    created_at: datetime

class HistoryEntry(BaseModel):
    id: str
    prompt: str
    prompt_type: str
    winning_provider: str
    winning_model: str
    total_tokens: int
    timestamp: datetime
    response_summary: str

# Auth helper functions
security = HTTPBearer()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": username,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

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
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
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

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Internal server error: {str(exc)}"
    logging.error(f"Global error handler caught: {error_msg}")
    return JSONResponse(
        status_code=500,
        content={"error": error_msg, "path": request.url.path}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - with error handling for serverless
try:
    frontend_dir = os.path.join(Path(__file__).parent, "frontend")
    if os.path.exists(frontend_dir):
        # Serve assets from /static
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
        logger.info(f"Static assets mounted from {frontend_dir}")
    else:
        logger.warning(f"Frontend directory not found at {frontend_dir}")
except Exception as e:
    logger.error(f"Failed to mount static files: {e}")

# Serve SPA index for root only (no catch-all that might mask /api/*)
@app.get("/")
async def serve_index():
    frontend_dir = os.path.join(Path(__file__).parent, "frontend")
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Multi-AI Webapp", "status": "running", "docs": "/docs"}

# Local run helper
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)

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
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

class TaskResponse(BaseModel):
    final_answer: str
    research_notes: str
    provider_results: List[ProviderResult]  # All providers' answers
    critic_report: str
    winning_provider: str
    winning_model: str
    prompt_type: str  # Detected prompt type
    total_tokens: int  # Total tokens across all providers

# --- Prompt-aware model selection ---
class PromptType(str, Enum):
    CODE = "code"
    DEBUG = "debug"
    MATH = "math"
    CREATIVE = "creative"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    FACTUAL = "factual"
    OTHER = "other"

def detect_prompt_type(prompt: str) -> PromptType:
    """Lightweight heuristic classification of the prompt to guide model selection.
    Purely local (no API calls)."""
    p = prompt.lower()
    # Code / debug
    code_keywords = ["code", "python", "javascript", "java", "c++", "bug", "stack trace", "traceback", "refactor", "unit test", "regex", "sql", "pandas", "error:"]
    if any(k in p for k in code_keywords):
        if "bug" in p or "error" in p or "stack" in p or "traceback" in p or "fix" in p:
            return PromptType.DEBUG
        return PromptType.CODE
    # Math / logic
    math_keywords = ["prove", "theorem", "derivative", "integral", "equation", "probability", "combinator", "logic puzzle", "optimize", "big-o"]
    if any(k in p for k in math_keywords):
        return PromptType.MATH
    # Creative writing
    creative_keywords = ["poem", "story", "creative", "tone", "style", "narrative", "script", "lyrics"]
    if any(k in p for k in creative_keywords):
        return PromptType.CREATIVE
    # Summarization / notes
    if "summarize" in p or "tl;dr" in p or "bullet" in p or "notes" in p:
        return PromptType.SUMMARIZE
    # Translation
    if "translate" in p or "in spanish" in p or "in french" in p or "in hindi" in p or "to english" in p:
        return PromptType.TRANSLATE
    # Factual / research-like
    factual_keywords = ["what is", "who is", "when was", "explain", "compare", "advantages", "disadvantages", "research", "sources"]
    if any(k in p for k in factual_keywords):
        return PromptType.FACTUAL
    return PromptType.OTHER

def _pick_if_available(provider: AIProvider, candidates: List[str]) -> str:
    """Return first candidate that exists in provider's allowed models, else default."""
    allowed = PROVIDER_CONFIGS[provider]["models"]
    for name in candidates:
        if name in allowed:
            return name
    return PROVIDER_CONFIGS[provider]["default_model"]

def select_model_for_prompt(provider: AIProvider, prompt_type: PromptType) -> str:
    """Choose a model for a provider given the prompt type, with safe fallback."""
    if provider == AIProvider.OPENAI:
        if prompt_type in {PromptType.CODE, PromptType.DEBUG, PromptType.MATH}:
            return _pick_if_available(provider, ["gpt-4-turbo", "gpt-4o"])  # stronger reasoning/code
        if prompt_type in {PromptType.SUMMARIZE, PromptType.TRANSLATE}:
            return _pick_if_available(provider, ["gpt-4o-mini", "gpt-3.5-turbo"])  # faster/cheaper
        if prompt_type == PromptType.CREATIVE:
            return _pick_if_available(provider, ["gpt-4o", "gpt-4-turbo"])  # better style/creativity
        return PROVIDER_CONFIGS[provider]["default_model"]
    elif provider == AIProvider.GEMINI:
        if prompt_type in {PromptType.CODE, PromptType.DEBUG, PromptType.MATH}:
            return _pick_if_available(provider, ["gemini-2.5-pro"])  # stronger reasoning/code
        if prompt_type in {PromptType.SUMMARIZE, PromptType.TRANSLATE}:
            return _pick_if_available(provider, ["gemini-2.0-flash", "gemini-2.5-flash-lite"])  # faster
        if prompt_type == PromptType.CREATIVE:
            return _pick_if_available(provider, ["gemini-flash-latest", "gemini-2.0-flash"])  # creative tasks
        return PROVIDER_CONFIGS[provider]["default_model"]
    elif provider == AIProvider.DEEPSEEK:
        if prompt_type in {PromptType.CODE, PromptType.DEBUG}:
            return _pick_if_available(provider, ["deepseek-coder"])  # code specialized
        # For other tasks, chat is fine
        return PROVIDER_CONFIGS[provider]["default_model"]
    else:
        return PROVIDER_CONFIGS[provider]["default_model"]

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
    async def chat(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = MAX_TOKENS) -> tuple[str, Dict[str, int]]:
        """Unified chat interface for all providers. Returns (response_text, token_usage)"""
        if self.provider == AIProvider.OPENAI:
            return await self._openai_chat(messages, temperature, max_tokens)
        elif self.provider == AIProvider.GEMINI:
            return await self._gemini_chat(messages, temperature, max_tokens)
        elif self.provider == AIProvider.DEEPSEEK:
            return await self._deepseek_chat(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _openai_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> tuple[str, Dict[str, int]]:
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
            
            # Extract token usage
            usage = data.get("usage", {})
            token_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            
            return data["choices"][0]["message"]["content"].strip(), token_info
    
    async def _gemini_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> tuple[str, Dict[str, int]]:
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
            
            # Extract token usage from Gemini response
            usage_metadata = data.get("usageMetadata", {})
            token_info = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0)
            }
            
            # Extract response text
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip(), token_info
            else:
                raise Exception(f"Gemini returned no candidates: {data}")
    
    async def _deepseek_chat(self, messages: List[Dict], temperature: float, max_tokens: int) -> tuple[str, Dict[str, int]]:
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
            
            # Extract token usage
            usage = data.get("usage", {})
            token_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            
            return data["choices"][0]["message"]["content"].strip(), token_info

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
    "You are an impartial AI Judge evaluating a competition between different AI providers.\n\n"
    "Your job:\n"
    "1. Analyze each AI provider's answer for accuracy, clarity, completeness, and usefulness\n"
    "2. Score each provider on a scale of 1-10 (10 = best) with brief rationale\n"
    "3. Identify specific strengths and weaknesses of each answer\n"
    "4. DO NOT favor any provider - be completely objective\n\n"
    "Format your response EXACTLY as:\n"
    "=== SCORING ===\n"
    "[Provider Name]: [Score]/10 - [Brief rationale]\n"
    "[Repeat for each provider]\n\n"
    "=== ANALYSIS ===\n"
    "[Detailed comparison of all answers, highlighting best elements from each]"
)

MERGER_SYSTEM = (
    "You are an expert AI tasked with creating the BEST possible answer by combining multiple AI responses.\n\n"
    "Your job:\n"
    "1. Review all the AI answers provided\n"
    "2. Extract the best elements from each (most accurate facts, clearest explanations, best examples)\n"
    "3. Fix any errors or inconsistencies\n"
    "4. Create a FINAL MERGED ANSWER that is better than any individual answer\n"
    "5. Make it clear, comprehensive, and well-structured\n\n"
    "Output ONLY the final merged answer - no meta-commentary."
)

async def researcher(task: str, ai_client: AIClient):
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nPlease provide research notes, bullets for facts, uncertainties, and 3 sources if available."}
    ]
    response, tokens = await ai_client.chat(messages, temperature=0.15, max_tokens=600)
    return response, tokens

async def writer_variant(research_notes: str, ai_client: AIClient, short=False):
    system = WRITER_SHORT_SYSTEM if short else WRITER_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Research notes:\n{research_notes}\n\nPlease produce the requested output."}
    ]
    temp = 0.6 if short else 0.5
    response, tokens = await ai_client.chat(messages, temperature=temp, max_tokens=700 if not short else 350)
    return response, tokens

async def critic(research_notes: str, provider_results: List[ProviderResult], ai_client: AIClient):
    """Single judge analyzes all provider results"""
    # Build comparison text
    comparison_text = "Research Notes:\n" + research_notes + "\n\n" + "="*60 + "\n\n"
    comparison_text += "PROVIDER ANSWERS TO EVALUATE:\n\n"
    
    for i, result in enumerate(provider_results, 1):
        if result.error:
            comparison_text += f"{i}. {result.provider.upper()} ({result.model}): [ERROR - {result.error}]\n\n"
        else:
            comparison_text += f"{i}. {result.provider.upper()} ({result.model}):\n{result.answer}\n\n{'─'*60}\n\n"
    
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": comparison_text}
    ]
    response, tokens = await ai_client.chat(messages, temperature=0.2, max_tokens=1000)
    return response, tokens

async def multi_judge_vote(research_notes: str, provider_results: List[ProviderResult], judge_clients: List[tuple]):
    """
    Have ALL available providers vote on the best answer.
    Returns: (aggregated_analysis, vote_scores, merged_answer)
    """
    vote_results = []
    
    # Each provider judges all the answers
    for judge_provider, judge_client in judge_clients:
        try:
            logger.info(f"Judge {judge_provider} evaluating all answers...")
            judge_result, judge_tokens = await critic(research_notes, provider_results, judge_client)
            vote_results.append({
                "judge": judge_provider,
                "evaluation": judge_result
            })
        except Exception as e:
            logger.error(f"Judge {judge_provider} failed: {e}")
            vote_results.append({
                "judge": judge_provider,
                "evaluation": f"[ERROR: {str(e)}]"
            })
    
    # Extract scores from each judge's evaluation
    scores = {}  # {provider: [scores from all judges]}
    for provider_result in provider_results:
        if not provider_result.error:
            scores[provider_result.provider] = []
    
    # Parse scores from each judge's evaluation
    for vote in vote_results:
        if "[ERROR" not in vote["evaluation"]:
            # Extract scores (looking for patterns like "Provider: 8/10" or "Provider: 8.5/10")
            for provider in scores.keys():
                # Look for score patterns
                pattern = rf"{provider}.*?(\d+(?:\.\d+)?)\s*/\s*10"
                match = re.search(pattern, vote["evaluation"], re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        scores[provider].append(score)
                    except:
                        pass
    
    # Calculate average scores
    avg_scores = {}
    for provider, score_list in scores.items():
        if score_list:
            avg_scores[provider] = sum(score_list) / len(score_list)
        else:
            avg_scores[provider] = 0
    
    # Determine winner (highest average score)
    winner = max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else "unknown"
    
    # Compile all judge evaluations
    aggregated_analysis = "=== MULTI-JUDGE VOTING RESULTS ===\n\n"
    for vote in vote_results:
        aggregated_analysis += f"🔍 JUDGE: {vote['judge'].upper()}\n"
        aggregated_analysis += vote['evaluation'] + "\n\n" + "="*80 + "\n\n"
    
    aggregated_analysis += "=== FINAL VOTE TALLY ===\n"
    for provider, avg_score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
        vote_count = len(scores.get(provider, []))
        aggregated_analysis += f"{'🏆 ' if provider == winner else ''}  {provider.upper()}: {avg_score:.1f}/10 (from {vote_count} judges)\n"
    
    aggregated_analysis += f"\n👑 WINNER: {winner.upper()} with average score of {avg_scores.get(winner, 0):.1f}/10\n"
    
    # Create merged answer using one of the judges
    if judge_clients:
        merger_client = judge_clients[0][1]  # Use first judge as merger
        comparison_for_merger = "\n\n".join([
            f"{r.provider.upper()}: {r.answer}" 
            for r in provider_results if not r.error
        ])
        
        merger_messages = [
            {"role": "system", "content": MERGER_SYSTEM},
            {"role": "user", "content": f"Research Notes:\n{research_notes}\n\nAI Answers:\n\n{comparison_for_merger}\n\nCreate the best merged answer:"}
        ]
        
        try:
            merged_answer, merger_tokens = await merger_client.chat(merger_messages, temperature=0.3, max_tokens=1200)
        except Exception as e:
            logger.error(f"Merger failed: {e}")
            # Fallback to winner's answer
            winner_result = next((r for r in provider_results if r.provider == winner), None)
            merged_answer = winner_result.answer if winner_result else provider_results[0].answer
    else:
        merged_answer = provider_results[0].answer if provider_results else "No answer available"
    
    return aggregated_analysis, avg_scores, winner, merged_answer

@app.post("/api/generate", response_model=TaskResponse)
async def generate(task: TaskRequest, current_user: str = Depends(verify_token)):
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
        
        # Classify prompt & log chosen type
        prompt_type = detect_prompt_type(task.prompt)
        logger.info(f"Detected prompt type: {prompt_type}")

        # Use the first available provider for research (they all see the same research)
        first_provider, first_key = available_providers[0]
        research_model = select_model_for_prompt(first_provider, prompt_type)
        research_client = AIClient(
            first_provider,
            first_key,
            research_model
        )
        
        # 1) Research phase (shared by all providers)
        logger.info(f"Research phase using {first_provider.value} with model {research_model}")
        research_notes, research_tokens = await researcher(task.prompt, research_client)
        
        # Track total tokens
        total_tokens = research_tokens.get("total_tokens", 0)
        
        # 2) Generate answers from ALL available providers concurrently
        logger.info(f"Running competition with {len(available_providers)} providers: {[p.value for p, _ in available_providers]}")
        
        async def get_provider_answer(provider: AIProvider, api_key: str) -> ProviderResult:
            """Get answer from a specific provider"""
            try:
                # Choose model dynamically based on prompt type
                model = select_model_for_prompt(provider, prompt_type)
                client = AIClient(provider, api_key, model)

                # Generate answer
                answer, tokens = await writer_variant(research_notes, client, short=False)

                return ProviderResult(
                    provider=provider.value,
                    model=model,
                    answer=answer,
                    error=None,
                    tokens_used=tokens.get("total_tokens", 0),
                    prompt_tokens=tokens.get("prompt_tokens", 0),
                    completion_tokens=tokens.get("completion_tokens", 0)
                )
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {str(e)}")
                return ProviderResult(
                    provider=provider.value,
                    model=PROVIDER_CONFIGS[provider]["default_model"],
                    answer="",
                    error=str(e)[:200],
                    tokens_used=0,
                    prompt_tokens=0,
                    completion_tokens=0
                )
        
        # Run all providers in parallel
        provider_tasks = [
            get_provider_answer(provider, api_key)
            for provider, api_key in available_providers
        ]
        provider_results = await asyncio.gather(*provider_tasks)
        
        # Calculate total tokens from all providers
        for result in provider_results:
            if result.tokens_used:
                total_tokens += result.tokens_used
        
        # Filter out results with errors for the critic (but keep them for display)
        successful_results = [r for r in provider_results if not r.error]
        
        if not successful_results:
            raise HTTPException(
                status_code=500,
                detail="All AI providers failed to generate answers. Please check your API keys and try again."
            )
        
        # 3) MULTI-JUDGE VOTING: Each available provider judges ALL answers
        logger.info(f"Multi-judge voting: {len(available_providers)} judges evaluating {len(successful_results)} answers")
        
        # Create judge clients (all available providers become judges)
        judge_clients = [
            (provider, AIClient(provider, api_key, select_model_for_prompt(provider, prompt_type)))
            for provider, api_key in available_providers
        ]
        
        # Run multi-judge voting
        critic_report, vote_scores, winning_provider, final_answer = await multi_judge_vote(
            research_notes, 
            provider_results, 
            judge_clients
        )
        
        # Get winning model
        winner_result = next((r for r in provider_results if r.provider == winning_provider), successful_results[0])
        winning_model = winner_result.model
        
        # Save to user history
        history_entry = HistoryEntry(
            id=str(uuid.uuid4()),
            prompt=task.prompt[:200] + "..." if len(task.prompt) > 200 else task.prompt,
            prompt_type=prompt_type.value,
            winning_provider=winning_provider,
            winning_model=winning_model,
            total_tokens=total_tokens,
            timestamp=datetime.utcnow(),
            response_summary=final_answer[:300] + "..." if len(final_answer) > 300 else final_answer
        )
        
        if current_user not in user_history_db:
            user_history_db[current_user] = []
        user_history_db[current_user].insert(0, history_entry.dict())  # Most recent first
        
        # Keep only last 50 entries per user
        user_history_db[current_user] = user_history_db[current_user][:50]
        
        return TaskResponse(
            final_answer=final_answer,
            research_notes=research_notes,
            provider_results=provider_results,
            critic_report=critic_report,
            winning_provider=winning_provider,
            winning_model=winning_model,
            prompt_type=prompt_type.value,
            total_tokens=total_tokens
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
    """Serve the dashboard (requires authentication)"""
    try:
        html_path = os.path.join(os.path.dirname(__file__), "frontend", "dashboard.html")
        if os.path.exists(html_path):
            return FileResponse(html_path)
        else:
            return {"message": "Multi-AI Webapp API", "status": "running", "docs": "/docs"}
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return {"message": "Multi-AI Webapp API", "status": "running", "docs": "/docs"}

@app.get("/auth.html")
async def auth_page():
    """Serve the authentication page"""
    try:
        html_path = os.path.join(os.path.dirname(__file__), "frontend", "auth.html")
        if os.path.exists(html_path):
            return FileResponse(html_path)
        else:
            raise HTTPException(status_code=404, detail="Auth page not found")
    except Exception as e:
        logger.error(f"Error serving auth page: {e}")
        raise HTTPException(status_code=500, detail="Error loading auth page")

@app.get("/dashboard.js")
async def dashboard_js():
    """Serve the dashboard JavaScript"""
    try:
        js_path = os.path.join(os.path.dirname(__file__), "frontend", "dashboard.js")
        if os.path.exists(js_path):
            return FileResponse(js_path, media_type="application/javascript")
        else:
            raise HTTPException(status_code=404, detail="Dashboard JS not found")
    except Exception as e:
        logger.error(f"Error serving dashboard JS: {e}")
        raise HTTPException(status_code=500, detail="Error loading dashboard JS")

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

# Authentication routes
@app.post("/api/auth/register")
async def register(user: UserRegister):
    # Check if username already exists
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email already exists
    for existing_user in users_db.values():
        if existing_user["email"] == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_data = {
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": datetime.utcnow()
    }
    users_db[user.username] = user_data
    user_history_db[user.username] = []
    
    # Create token
    token = create_access_token(user.username)
    
    return {
        "user": UserResponse(
            username=user.username,
            email=user.email,
            created_at=user_data["created_at"]
        ),
        "token": token
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    # Check if user exists
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user_data = users_db[user.username]
    
    # Verify password
    if not verify_password(user.password, user_data["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create token
    token = create_access_token(user.username)
    
    return {
        "user": UserResponse(
            username=user_data["username"],
            email=user_data["email"],
            created_at=user_data["created_at"]
        ),
        "token": token
    }

@app.get("/api/auth/me")
async def get_current_user(current_user: str = Depends(verify_token)):
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = users_db[current_user]
    return UserResponse(
        username=user_data["username"],
        email=user_data["email"],
        created_at=user_data["created_at"]
    )

@app.get("/api/auth/history")
async def get_user_history(current_user: str = Depends(verify_token)):
    return user_history_db.get(current_user, [])

# Optional hooks for LangChain/AutoGen (placeholders)
# If you want to integrate LangChain or AutoGen orchestration, add that logic below.
# Example: call a LangChain LLMChain with prompts for each role instead of direct openai_chat.
