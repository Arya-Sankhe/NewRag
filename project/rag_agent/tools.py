from typing import List, Dict
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager
from core.chat_interface import image_tracker

class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()
    
    def _search_child_chunks(self, query: str, k: int) -> List[dict]:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            k: Number of results to return
        """
        try:
            # Note: Removed score_threshold as it was filtering out valid results
            # OpenAI embeddings cosine similarity often returns scores < 0.7
            results = self.collection.similarity_search(query, k=k)
            
            print(f"🔍 Search query: '{query}' → Found {len(results)} results")
            
            if not results:
                print("   ⚠️ No results found - check if documents are indexed")
                return []
            
            return [
                {
                    "content": doc.page_content,
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "source": doc.metadata.get("source", "")
                }
                for doc in results
            ]
        except Exception as e:
            print(f"❌ Error searching child chunks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _retrieve_parent_chunks(self, parent_ids: List[str]) -> List[dict]:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
            
        Returns:
            List of parent chunk dicts with content (image selection handled by CLIP separately)
        """
        results = self.parent_store_manager.load_many(parent_ids)
        
        # Track parent IDs for post-response image injection
        for parent_id in parent_ids:
            image_tracker.track(parent_id)
            print(f"📷 Tracked parent_id for images: {parent_id}")
        
        # Process results - return clean content without image references
        # Image selection is now handled by CLIP scoring, not the LLM
        processed_results = []
        for result in results:
            clean_result = {
                "content": result.get("content", ""),
                "parent_id": result.get("parent_id", ""),
                "source": result.get("metadata", {}).get("source", ""),
            }
            processed_results.append(clean_result)
        
        return processed_results
    
    def _format_images_for_display(self, images: List[Dict]) -> List[Dict]:
        """
        Format images with proper data URLs for display.
        
        Args:
            images: List of image metadata dicts with base64_data
            
        Returns:
            List of formatted image dicts with data_url
        """
        formatted = []
        for img in images:
            base64_data = img.get("base64_data", "")
            if not base64_data:
                continue
            
            mime_type = img.get("mime_type", "image/png")
            
            # Ensure no duplicate data: prefix
            if base64_data.startswith("data:"):
                data_url = base64_data
            else:
                data_url = f"data:{mime_type};base64,{base64_data}"
            
            formatted.append({
                "image_id": img.get("image_id", ""),
                "data_url": data_url,
                "caption": img.get("caption", "") or img.get("description", ""),
                "page_number": img.get("page_number")
            })
        
        return formatted
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]