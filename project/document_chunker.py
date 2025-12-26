import os
import json
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChuncker:
    def __init__(self):
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE, 
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)
        
        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())
        
        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)
        
        # Load images metadata if available
        images_metadata = self._load_images_metadata(doc_path)
        
        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path, images_metadata)
        return all_parent_chunks, all_child_chunks
    
    def _load_images_metadata(self, md_path: Path) -> list:
        """Load images metadata from JSON file if it exists."""
        images_json_path = md_path.parent / f"{md_path.stem}_images.json"
        if images_json_path.exists():
            try:
                with open(images_json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load images metadata: {e}")
        return []

    def __merge_small_parents(self, chunks):
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v

            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged

    def __split_large_parents(self, chunks):
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                sub_chunks = splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

    def __clean_small_chunks(self, chunks):
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path, images_metadata=None):
        """
        Create child chunks from parent chunks and link images.
        
        Args:
            all_parent_pairs: List to append (parent_id, parent_chunk) tuples
            all_child_chunks: List to append child chunk documents
            parent_chunks: List of parent chunk documents
            doc_path: Path to source document
            images_metadata: List of image metadata dicts from Docling
        """
        images_metadata = images_metadata or []
        
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({
                "source": str(doc_path.stem) + ".pdf", 
                "parent_id": parent_id
            })
            
            # Link images to this parent chunk
            # Strategy: Associate images with chunks based on order or page number
            if images_metadata:
                # Try to get page numbers from images
                chunk_images = self._get_images_for_chunk(i, len(parent_chunks), images_metadata)
                if chunk_images:
                    p_chunk.metadata["ocr_images"] = chunk_images
            
            all_parent_pairs.append((parent_id, p_chunk))
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))
    
    def _get_images_for_chunk(self, chunk_index: int, total_chunks: int, images_metadata: list) -> list:
        """
        Get images that should be associated with a particular chunk.
        
        Uses a distribution strategy based on page numbers if available,
        otherwise distributes images evenly across chunks.
        """
        if not images_metadata:
            return []
        
        # Collect page numbers from images
        page_numbers = set()
        for img in images_metadata:
            if img.get("page_number") is not None:
                page_numbers.add(img["page_number"])
        
        if page_numbers:
            # If images have page numbers, use page-based distribution
            # Estimate which pages this chunk covers
            max_page = max(page_numbers)
            pages_per_chunk = max(1, max_page / total_chunks)
            chunk_start_page = int(chunk_index * pages_per_chunk) + 1
            chunk_end_page = int((chunk_index + 1) * pages_per_chunk) + 1
            
            chunk_images = [
                img for img in images_metadata
                if img.get("page_number") is not None 
                and chunk_start_page <= img["page_number"] <= chunk_end_page
            ]
            return chunk_images
        else:
            # Distribute images evenly if no page numbers
            images_per_chunk = max(1, len(images_metadata) // total_chunks)
            start_idx = chunk_index * images_per_chunk
            end_idx = start_idx + images_per_chunk if chunk_index < total_chunks - 1 else len(images_metadata)
            return images_metadata[start_idx:end_idx]