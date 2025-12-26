import os
import json
import config
from pathlib import Path
import glob
from typing import Tuple, List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pdf_to_markdown(pdf_path: str, output_dir: str) -> Tuple[Path, List[Dict]]:
    """
    Convert PDF to markdown with optional image extraction.
    
    Uses Docling if enabled in config, otherwise falls back to PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save markdown file
        
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
            
            markdown_text, images_metadata = parser.convert(pdf_path)
            md_path.write_text(markdown_text, encoding='utf-8')
            
            # Save images metadata as JSON
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