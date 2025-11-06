"""
Vercel serverless function handler for FastAPI
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from mangum import Mangum

# Vercel handler
handler = Mangum(app)
