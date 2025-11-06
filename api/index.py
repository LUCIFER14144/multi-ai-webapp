"""
Vercel serverless function handler for FastAPI
"""
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import app
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Change working directory to parent
os.chdir(parent_dir)

try:
    from app.main import app
    from mangum import Mangum
    
    # Vercel handler
    handler = Mangum(app, lifespan="off")
except Exception as e:
    # If import fails, create a simple error handler
    import traceback
    error_details = traceback.format_exc()
    
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    app = FastAPI()
    
    @app.get("/")
    @app.post("/")
    @app.get("/{path:path}")
    @app.post("/{path:path}")
    async def error_handler(path: str = ""):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Application failed to start",
                "details": str(e),
                "traceback": error_details
            }
        )
    
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
