from langchain_core.messages import HumanMessage
from db.parent_store_manager import ParentStoreManager
from rag_agent.image_scorer import score_images_for_query
import config


class ImageTracker:
    """Tracks parent IDs retrieved during a query for post-response image injection."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.retrieved_parent_ids = set()
        return cls._instance
    
    def track(self, parent_id: str):
        self.retrieved_parent_ids.add(parent_id)
    
    def get_and_clear(self) -> set:
        ids = self.retrieved_parent_ids.copy()
        self.retrieved_parent_ids.clear()
        return ids


# Global tracker instance
image_tracker = ImageTracker()


class ChatInterface:
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.parent_store = ParentStoreManager()
        
    def chat(self, message, history):
        if not self.rag_system.agent_graph:
            return "⚠️ System not initialized!"
        
        # Clear any previous tracked IDs
        image_tracker.get_and_clear()
            
        try:
            # Run the agent
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            
            # Get LLM response
            response_text = result["messages"][-1].content
            
            # Get images from retrieved chunks and score with CLIP
            retrieved_ids = image_tracker.get_and_clear()
            if retrieved_ids:
                images_markdown = self._get_relevant_images(message, retrieved_ids)
                if images_markdown:
                    response_text += images_markdown
            
            return response_text
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _get_relevant_images(self, query: str, parent_ids: set) -> str:
        """
        Get relevant images using CLIP semantic scoring.
        
        Args:
            query: User query for relevance matching
            parent_ids: Parent IDs to fetch images from
            
        Returns:
            HTML string with relevant images, or empty string
        """
        # Collect all images from retrieved parents
        all_images = []
        
        for parent_id in parent_ids:
            parent_data = self.parent_store.load(parent_id)
            if not parent_data:
                continue
            
            ocr_images = parent_data.get("metadata", {}).get("ocr_images", [])
            print(f"   📷 Parent {parent_id}: found {len(ocr_images)} images in metadata")
            
            for img in ocr_images:
                # Support both file-based (image_path) and legacy (base64_data)
                if img.get("image_path") or img.get("base64_data"):
                    img_copy = img.copy()
                    img_copy["parent_id"] = parent_id
                    all_images.append(img_copy)
        
        if not all_images:
            print("   📷 No images found in retrieved chunks")
            return ""
        
        print(f"   📷 Scoring {len(all_images)} images with CLIP...")
        
        # Score images with CLIP
        relevant_images = score_images_for_query(query, all_images)
        
        if not relevant_images:
            print("   📷 No images passed relevance threshold")
            return ""
        
        print(f"   ✓ Found {len(relevant_images)} relevant images")
        
        # Format as HTML
        return self._format_images_html(relevant_images)
    
    def _format_images_html(self, images: list) -> str:
        """Format scored images as HTML with API URLs."""
        
        html = "\n\n---\n\n**📸 Related Images:**\n\n"
        images_added = 0
        
        for img in images:
            # Determine image source
            image_path = img.get("image_path", "")
            base64_data = img.get("base64_data", "")
            mime_type = img.get("mime_type", "image/png")
            
            # Prefer file path (new system), fallback to base64 (legacy)
            if image_path:
                # Use API URL - strip "images/" prefix if present
                if image_path.startswith("images/"):
                    image_path = image_path[7:]  # Remove "images/"
                img_url = f"/api/v1/images/{image_path}"
            elif base64_data:
                # Legacy base64 fallback
                if base64_data.startswith("data:"):
                    img_url = base64_data
                else:
                    img_url = f"data:{mime_type};base64,{base64_data}"
            else:
                print(f"   ⚠️ Skipping image with no path or data: {img.get('image_id', 'unknown')}")
                continue
            
            # Caption and score
            caption = img.get("caption", "") or img.get("vlm_caption", "") or img.get("description", "")
            score = img.get("relevance_score", 0)
            page_num = img.get("page_number")
            
            # Build caption text
            if caption:
                caption_text = f"*{caption}* ({score:.0%})"
            elif page_num:
                caption_text = f"*Page {page_num}* ({score:.0%})"
            else:
                caption_text = f"({score:.0%})"
            
            # Use HTML img tag
            html += f"{caption_text}\n\n"
            html += f'<img src="{img_url}" alt="Related image" style="max-width: 100%; max-height: 400px; border-radius: 8px; margin: 8px 0;" />\n\n'
            images_added += 1
        
        # Return empty if no valid images
        if images_added == 0:
            print("   ⚠️ No images had valid path or data")
            return ""
        
        print(f"   ✓ Added {images_added} images to response")
        return html
    
    def clear_session(self):
        self.rag_system.reset_thread()
        image_tracker.get_and_clear()