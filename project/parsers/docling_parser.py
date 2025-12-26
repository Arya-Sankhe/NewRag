"""
Docling PDF Parser Module

Advanced PDF parser using Docling for OCR and image extraction.
Converts PDFs to markdown with full image metadata extraction.
"""

import base64
import io
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Optional imports - guarded for when dependencies are not installed
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    DocumentConverter = None
    PdfFormatOption = None
    InputFormat = None
    PdfPipelineOptions = None


class DoclingPDFParser:
    """
    Advanced PDF parser using Docling for OCR and image extraction.
    
    Features:
    - OCR-based text extraction
    - Image extraction with metadata
    - Caption generation for images
    - Bounding box preservation
    - Base64 encoding for storage
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        generate_page_images: bool = False,
        generate_picture_images: bool = True,
        images_scale: float = 2.0,
        do_picture_description: bool = False
    ):
        """
        Initialize Docling converter with OCR and image extraction options.
        
        Args:
            enable_ocr: Enable OCR for scanned documents
            generate_page_images: Generate full page images (memory intensive)
            generate_picture_images: Extract embedded images from PDF
            images_scale: Scale factor for extracted images (higher = better quality)
            do_picture_description: Generate AI descriptions for images (requires VLM)
            
        Raises:
            ImportError: If Docling is not installed
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling docling-core"
            )
        
        self.enable_ocr = enable_ocr
        self.images_scale = images_scale
        
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(
            do_ocr=enable_ocr,
            generate_page_images=generate_page_images,
            generate_picture_images=generate_picture_images,
            images_scale=images_scale,
            do_picture_description=do_picture_description,
        )
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    
    def convert(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """
        Convert PDF to markdown with image extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            tuple: (markdown_text, images_metadata)
                images_metadata: List of dicts containing:
                    - image_id: Unique identifier
                    - page_number: Page where image appears
                    - base64_data: Base64 encoded image (raw, no data: prefix)
                    - caption: Caption text if available
                    - bbox: Bounding box coordinates [x0, y0, x1, y1]
                    - mime_type: Image MIME type (e.g., image/png)
        """
        print(f"📄 Converting PDF with Docling: {Path(pdf_path).name}")
        
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        # Export to markdown
        markdown_text = doc.export_to_markdown()
        
        # Extract images with metadata
        images_metadata = self._extract_all_images(doc, pdf_path)
        
        print(f"   ✓ Extracted {len(images_metadata)} images")
        
        return markdown_text, images_metadata
    
    def _extract_all_images(self, doc, pdf_path: str) -> List[Dict]:
        """Extract all images from the document with metadata."""
        images_metadata = []
        doc_stem = Path(pdf_path).stem
        
        # Check if document has pictures
        if not hasattr(doc, 'pictures') or not doc.pictures:
            return images_metadata
        
        for idx, picture in enumerate(doc.pictures):
            try:
                image_data = self._extract_image_metadata(picture, idx, doc, doc_stem)
                if image_data:
                    images_metadata.append(image_data)
            except Exception as e:
                print(f"   ⚠️ Warning: Could not extract image {idx}: {e}")
                continue
        
        return images_metadata
    
    def _extract_image_metadata(self, picture, index: int, doc, doc_stem: str) -> Optional[Dict]:
        """
        Extract comprehensive metadata from a picture object.
        
        Args:
            picture: Docling picture object
            index: Image index in document
            doc: Parent document object
            doc_stem: Document filename without extension
            
        Returns:
            Dict with image metadata or None if extraction failed
        """
        # Get image data
        if not hasattr(picture, 'image') or picture.image is None:
            return None
        
        # Extract base64 data
        base64_data, mime_type = self._image_to_base64(picture.image)
        if not base64_data:
            return None
        
        # Extract caption
        caption = ""
        if hasattr(picture, 'caption_text'):
            try:
                caption = picture.caption_text(doc=doc) or ""
            except Exception:
                pass
        
        # Extract description from annotations if available
        description = ""
        if hasattr(picture, 'annotations') and picture.annotations:
            for annotation in picture.annotations:
                if hasattr(annotation, 'text') and annotation.text:
                    description = annotation.text
                    break
        
        # Extract page number and bounding box
        page_number = None
        bbox = None
        if hasattr(picture, 'prov') and picture.prov:
            prov = picture.prov[0] if isinstance(picture.prov, list) else picture.prov
            if hasattr(prov, 'page_no'):
                page_number = prov.page_no
            if hasattr(prov, 'bbox'):
                try:
                    bbox = list(prov.bbox.as_tuple()) if hasattr(prov.bbox, 'as_tuple') else None
                except Exception:
                    pass
        
        # Generate unique image ID
        image_id = f"{doc_stem}_img_{index}"
        
        return {
            "image_id": image_id,
            "page_number": page_number,
            "base64_data": base64_data,  # Raw base64, no data: prefix
            "mime_type": mime_type,
            "caption": caption,
            "description": description,
            "bbox": bbox,
            "self_ref": str(picture.self_ref) if hasattr(picture, 'self_ref') else None
        }
    
    def _image_to_base64(self, image_obj) -> Tuple[Optional[str], str]:
        """
        Convert Docling image object to base64 string.
        
        Args:
            image_obj: Docling image object (may be PIL Image or have uri/pil_image attr)
            
        Returns:
            Tuple of (base64_string, mime_type) or (None, "") if failed
        """
        try:
            pil_image = None
            mime_type = "image/png"  # Default
            
            # Try different ways to get PIL image
            if hasattr(image_obj, 'pil_image') and image_obj.pil_image is not None:
                pil_image = image_obj.pil_image
            elif hasattr(image_obj, 'uri') and image_obj.uri:
                # URI might be a data URI or file path
                uri = str(image_obj.uri)
                if uri.startswith('data:'):
                    # Extract base64 from data URI
                    if ',' in uri:
                        header, data = uri.split(',', 1)
                        if 'image/jpeg' in header:
                            mime_type = "image/jpeg"
                        elif 'image/png' in header:
                            mime_type = "image/png"
                        return data, mime_type
                elif os.path.exists(uri):
                    pil_image = Image.open(uri)
            elif isinstance(image_obj, Image.Image):
                pil_image = image_obj
            
            if pil_image is None:
                return None, ""
            
            # Determine format based on image mode
            img_format = "PNG"
            if pil_image.mode == "RGB":
                img_format = "JPEG"
                mime_type = "image/jpeg"
            elif pil_image.mode == "RGBA":
                img_format = "PNG"
                mime_type = "image/png"
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format=img_format, quality=85)
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            return base64_data, mime_type
            
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to convert image to base64: {e}")
            return None, ""
    
    def convert_and_save(
        self, 
        pdf_path: str, 
        output_dir: str,
        save_images_json: bool = True
    ) -> Tuple[Path, List[Dict]]:
        """
        Convert PDF and save markdown + images metadata.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save markdown and images JSON
            save_images_json: Whether to save images metadata as JSON
            
        Returns:
            Tuple of (markdown_path, images_metadata)
        """
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc_stem = Path(pdf_path).stem
        
        # Convert PDF
        markdown_text, images_metadata = self.convert(pdf_path)
        
        # Save markdown
        md_path = output_dir / f"{doc_stem}.md"
        md_path.write_text(markdown_text, encoding='utf-8')
        
        # Save images metadata
        if save_images_json and images_metadata:
            images_json_path = output_dir / f"{doc_stem}_images.json"
            with open(images_json_path, 'w', encoding='utf-8') as f:
                json.dump(images_metadata, f, ensure_ascii=False, indent=2)
        
        return md_path, images_metadata


# Fallback parser using PyMuPDF for when Docling is disabled
class PyMuPDFParser:
    """Fallback PDF parser using PyMuPDF (no image extraction)."""
    
    def convert(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Convert PDF to markdown (no image extraction)."""
        import pymupdf
        import pymupdf4llm
        
        doc = pymupdf.open(pdf_path)
        md = pymupdf4llm.to_markdown(
            doc, 
            header=False, 
            footer=False, 
            page_separators=True, 
            ignore_images=True, 
            write_images=False, 
            image_path=None
        )
        md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        
        # No image extraction with PyMuPDF fallback
        return md_cleaned, []
    
    def convert_and_save(
        self, 
        pdf_path: str, 
        output_dir: str,
        save_images_json: bool = True
    ) -> Tuple[Path, List[Dict]]:
        """Convert and save markdown (no images)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        doc_stem = Path(pdf_path).stem
        markdown_text, images_metadata = self.convert(pdf_path)
        
        md_path = output_dir / f"{doc_stem}.md"
        md_path.write_text(markdown_text, encoding='utf-8')
        
        return md_path, images_metadata


def get_parser(use_docling: bool = True):
    """
    Factory function to get the appropriate parser.
    
    Args:
        use_docling: If True, use Docling parser; otherwise use PyMuPDF fallback
        
    Returns:
        Parser instance
    """
    if use_docling:
        try:
            return DoclingPDFParser()
        except ImportError as e:
            print(f"⚠️ Docling not available, falling back to PyMuPDF: {e}")
            return PyMuPDFParser()
    return PyMuPDFParser()
