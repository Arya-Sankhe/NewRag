"""
Image serving API routes.

Provides endpoint for serving images stored on disk.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import mimetypes

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
    Example: /api/v1/images/ORION_MANUAL/img_0.png
    
    Args:
        doc_stem: Document name (folder name)
        filename: Image filename (e.g., img_0.png)
        
    Returns:
        FileResponse with the image
    """
    # Validate path components (prevent directory traversal)
    if ".." in doc_stem or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    if "/" in doc_stem or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Build full path
    image_path = IMAGES_BASE_PATH / doc_stem / filename
    
    # Check if file exists
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    
    # Get MIME type
    mime_type = get_mime_type(filename)
    
    return FileResponse(
        path=str(image_path),
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=86400"  # Cache for 1 day
        }
    )
