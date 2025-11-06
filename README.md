# Multi-AI Web App (FastAPI) - Multi-Provider Edition

This project provides a web application that runs a 3-AI pipeline (Researcher → Writer(s) → Critic) using multiple AI providers: **OpenAI (ChatGPT)**, **Google Gemini**, and **DeepSeek**. Users provide their own API keys through the frontend interface.

## ✨ Features
- **Multi-Provider Support**: OpenAI, Google Gemini, DeepSeek
- **User-Provided API Keys**: No server-side key storage
- **3-AI Pipeline**: Researcher → Writer(s) → Critic workflow
- **Async Processing**: Fast, concurrent AI calls
- **Retry Logic**: Robust error handling
- **Modern Frontend**: Clean, responsive interface

## 📁 Project Structure
- `app/main.py` — FastAPI server with multi-provider AI client
- `app/frontend/index.html` + `app/frontend/app.js` — Modern frontend interface
- `Dockerfile` — Container configuration
- `requirements.txt` — Python dependencies

## 🚀 Quick Start

### Local Development
1. **Install dependencies:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

2. **Run the application:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Open your browser:**
   - Go to `http://localhost:8000`
   - Open `app/frontend/index.html` directly in browser
   - Or serve frontend files with any static server

## 🐳 Docker Deployment

```bash
# Build image
docker build -t multi-ai-app:latest .

# Run container (no API keys needed in environment)
docker run -p 8000:8000 multi-ai-app:latest
```

## 🔑 Supported AI Providers

### OpenAI (ChatGPT)
- **Models**: gpt-4o-mini, gpt-4o, gpt-3.5-turbo, gpt-4-turbo
- **API Key**: Get from [OpenAI Platform](https://platform.openai.com/)

### Google Gemini
- **Models**: gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro
- **API Key**: Get from [Google AI Studio](https://aistudio.google.com/)

### DeepSeek
- **Models**: deepseek-chat, deepseek-coder
- **API Key**: Get from [DeepSeek Platform](https://platform.deepseek.com/)

## 🛡️ Security Features
- API keys never stored on server
- Input validation and sanitization
- Rate limiting and retry logic
- CORS configuration
- Error handling with proper HTTP status codes

## 📊 API Endpoints
- `POST /api/generate` - Run AI pipeline
- `GET /api/providers` - Get available providers/models
- `GET /health` - Health check

## 🔧 Production Considerations
- Implement authentication/authorization
- Add request rate limiting per user
- Set up proper logging and monitoring
- Configure CORS for specific domains
- Use secrets management for sensitive data
- Add request/response caching
  - Add authentication, rate limiting, logging, and observability.
  - Replace prompt strings with external prompt templates or use LangChain/AutoGen orchestration.
  - Add streaming for faster UX (OpenAI streaming / SSE).
