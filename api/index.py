"""
Minimal Vercel serverless entry for FastAPI.
Expose the FastAPI `app` directly — no Mangum, no cwd/env mutations.
"""

from pathlib import Path
import sys

# Ensure project root is importable
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import FastAPI app from our package
from app.main import app as app  # Vercel looks for `app`
    
