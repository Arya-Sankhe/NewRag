"""
Image serving API routes.

Provides endpoint for serving images stored on disk.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import mimetypes
from urllib.parse import unquote

router = APIRouter()

# Base path for images (mounted volume in Docker)
IMAGES_BASE_PATH = Path("/app/project/images")


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "image/png"


@router.get("/{doc_stem}/{filename}")
async def serve_image(doc_stem: str, filename: str):
    """
    Serve an image file from the images directory.
    
    Path: /api/v1/images/{doc_stem}/{filename}
    Example: /api/v1/images/ORION%20MANUAL/img_0.png
    
    Args:
        doc_stem: Document name (folder name) - URL decoded automatically by FastAPI
        filename: Image filename (e.g., img_0.png)
        
    Returns:
        FileResponse with the image
    """
    # URL decode the path components (handles %20 for spaces, etc.)
    doc_stem = unquote(doc_stem)
    filename = unquote(filename)
    
    print(f"📷 Image request: doc_stem='{doc_stem}', filename='{filename}'")
    
    # Validate path components (prevent directory traversal)
    if ".." in doc_stem or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Build full path
    image_path = IMAGES_BASE_PATH / doc_stem / filename
    
    print(f"   Looking for: {image_path}")
    
    # Check if file exists
    if not image_path.exists():
        print(f"   ❌ Not found: {image_path}")
        # List what's in the directory for debugging
        if IMAGES_BASE_PATH.exists():
            dirs = list(IMAGES_BASE_PATH.iterdir())
            print(f"   Available dirs: {[d.name for d in dirs]}")
        raise HTTPException(status_code=404, detail=f"Image not found: {doc_stem}/{filename}")
    
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    
    print(f"   ✓ Found: {image_path}")
    
    # Get MIME type
    mime_type = get_mime_type(filename)
    
    return FileResponse(
        path=str(image_path),
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=86400"  # Cache for 1 day
        }
    )
