import os
import json
import config
from pathlib import Path
import glob
from typing import Tuple, List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pdf_to_markdown(pdf_path: str, output_dir: str, enable_vlm: bool = False) -> Tuple[Path, List[Dict]]:
    """
    Convert PDF to markdown with optional image extraction and VLM captions.
    
    Uses Docling if enabled in config, otherwise falls back to PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save markdown file
        enable_vlm: If True, use VLM for enhanced image captions
        
    Returns:
        Tuple of (markdown_path, images_metadata)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc_stem = Path(pdf_path).stem
    md_path = output_dir / f"{doc_stem}.md"
    images_metadata = []
    
    if config.ENABLE_DOCLING:
        try:
            from parsers.docling_parser import DoclingPDFParser
            
            parser = DoclingPDFParser(
                enable_ocr=config.DOCLING_OCR_ENABLED,
                images_scale=config.DOCLING_IMAGE_SCALE,
                do_picture_description=config.DOCLING_GENERATE_CAPTIONS
            )
            
            # Use convert_and_save to save images to disk and get image_path in metadata
            md_path, images_metadata = parser.convert_and_save(pdf_path, str(output_dir))
            
            # Generate VLM captions if enabled
            if enable_vlm and images_metadata:
                images_metadata = _add_vlm_captions(images_metadata)
                # Re-save JSON with VLM captions
                if images_metadata:
                    images_json_path = output_dir / f"{doc_stem}_images.json"
                    with open(images_json_path, 'w', encoding='utf-8') as f:
                        json.dump(images_metadata, f, ensure_ascii=False, indent=2)
            
            return md_path, images_metadata
            
        except ImportError as e:
            print(f"⚠️ Docling not available, falling back to PyMuPDF: {e}")
        except Exception as e:
            print(f"⚠️ Docling failed, falling back to PyMuPDF: {e}")
    
    # Fallback to PyMuPDF
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
    md_path.write_bytes(md_cleaned.encode('utf-8'))
    
    return md_path, []


def _add_vlm_captions(images_metadata: List[Dict]) -> List[Dict]:
    """
    Add VLM-generated captions to images using OpenAI's vision model.
    
    Supports both file-based images (image_path) and legacy base64_data.
    
    Args:
        images_metadata: List of image metadata dicts
        
    Returns:
        Updated images_metadata with vlm_caption field
    """
    import base64
    from pathlib import Path
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Use GPT-4o-mini for cost-effective vision
        vlm = ChatOpenAI(model="gpt-4o-mini", max_tokens=150)
        
        print(f"   🧠 Generating VLM captions for {len(images_metadata)} images...")
        
        # Get project root for resolving relative paths
        project_root = Path(config.__file__).parent
        
        for i, img in enumerate(images_metadata):
            image_url = None
            mime_type = img.get("mime_type", "image/png")
            
            # Try to get image data - prefer file path, fallback to base64
            image_path = img.get("image_path", "")
            base64_data = img.get("base64_data", "")
            
            if image_path:
                # Load image from file and convert to base64
                try:
                    full_path = project_root / image_path
                    if full_path.exists():
                        with open(full_path, 'rb') as f:
                            image_bytes = f.read()
                        b64_data = base64.b64encode(image_bytes).decode('utf-8')
                        image_url = f"data:{mime_type};base64,{b64_data}"
                    else:
                        print(f"      ⚠️ Image file not found: {full_path}")
                        continue
                except Exception as e:
                    print(f"      ⚠️ Failed to load image {i+1}: {e}")
                    continue
            elif base64_data:
                # Legacy base64 support
                if base64_data.startswith("data:"):
                    image_url = base64_data
                else:
                    image_url = f"data:{mime_type};base64,{base64_data}"
            else:
                print(f"      ⚠️ No image data for image {i+1}")
                continue
            
            # Retry logic with exponential backoff for rate limits
            max_retries = 3
            retry_delay = 0.5  # Start with 500ms
            
            for attempt in range(max_retries):
                try:
                    # Create vision message
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Describe this image in 1-2 sentences. Focus on what the image shows (charts, diagrams, photos, etc.) and any key information visible. Be concise."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    )
                    
                    response = vlm.invoke([message])
                    vlm_caption = response.content.strip()
                    
                    img["vlm_caption"] = vlm_caption
                    print(f"      ✓ Image {i+1}: {vlm_caption[:50]}...")
                    
                    # Add delay between successful calls to avoid rate limits
                    import time
                    time.sleep(0.3)  # 300ms between calls
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "rate_limit" in error_str.lower():
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"      ⏳ Rate limited, waiting {wait_time:.1f}s...")
                            import time
                            time.sleep(wait_time)
                            continue
                    # Non-rate-limit error or final attempt
                    print(f"      ⚠️ VLM failed for image {i+1}: {e}")
                    img["vlm_caption"] = ""
                    break
        
        print(f"   ✓ VLM captions complete")
        return images_metadata
        
    except ImportError as e:
        print(f"   ⚠️ VLM not available (langchain_openai not installed): {e}")
        return images_metadata
    except Exception as e:
        print(f"   ⚠️ VLM caption generation failed: {e}")
        return images_metadata


def pdfs_to_markdowns(path_pattern: str, overwrite: bool = False) -> Tuple[int, int]:
    """
    Convert multiple PDFs to markdown.
    
    Args:
        path_pattern: Glob pattern for PDF files
        overwrite: Whether to overwrite existing markdown files
        
    Returns:
        Tuple of (converted_count, skipped_count)
    """
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    skipped = 0
    
    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            try:
                pdf_to_markdown(str(pdf_path), str(output_dir))
                converted += 1
            except Exception as e:
                print(f"❌ Error converting {pdf_path.name}: {e}")
                skipped += 1
        else:
            skipped += 1
    
    return converted, skipped