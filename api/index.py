"""
Vercel serverless function handler for FastAPI
"""
import sys
import os
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vercel-handler")

# Add parent directory to path so we can import app
try:
    # Get absolute paths
    handler_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(handler_dir)
    
    # Add to Python path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added {parent_dir} to Python path")
    
    # Set up environment
    os.environ["PYTHONPATH"] = parent_dir
    os.environ["APP_ROOT"] = parent_dir
    
    # Log critical paths
    logger.info(f"Handler directory: {handler_dir}")
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
except Exception as e:
    logger.error(f"Failed to set up paths: {traceback.format_exc()}")

try:
    from app.main import app
    from mangum import Mangum
    
    # Log environment for debugging
    logger.info("Python version: " + sys.version)
    logger.info("Current working directory: " + os.getcwd())
    logger.info("Contents of current directory: " + str(os.listdir(".")))
    logger.info("Contents of app directory: " + str(os.listdir("app")))
    
    # Create handler with debug mode
    handler = Mangum(
        app,
        lifespan="off",
        api_gateway_base_path="/api",
        strip_base_path=False,
        debug=True
    )
    logger.info("Mangum handler created successfully with debug mode enabled")
    
except Exception as e:
    # If import fails, create a simple error handler
    import traceback
    error_details = traceback.format_exc()
    logger.error(f"Failed to create handler: {str(e)}\n{error_details}")
    
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
                "traceback": error_details,
                "cwd": os.getcwd(),
                "path": path,
                "sys_path": sys.path,
                "dir_contents": {
                    "root": os.listdir(".") if os.path.exists(".") else [],
                    "app": os.listdir("app") if os.path.exists("app") else []
                }
            }
        )
    
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
