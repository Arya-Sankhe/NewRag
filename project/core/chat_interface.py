from langchain_core.messages import HumanMessage
from db.parent_store_manager import ParentStoreManager


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
            
            # Get images for retrieved parent IDs and append to response
            retrieved_ids = image_tracker.get_and_clear()
            if retrieved_ids:
                images_markdown = self._get_images_markdown(retrieved_ids)
                if images_markdown:
                    response_text += images_markdown
            
            return response_text
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _get_images_markdown(self, parent_ids: set) -> str:
        """Fetch images for parent IDs and format as markdown."""
        all_images = []
        
        for parent_id in parent_ids:
            parent_data = self.parent_store.load(parent_id)
            if not parent_data:
                continue
            
            ocr_images = parent_data.get("metadata", {}).get("ocr_images", [])
            for img in ocr_images:
                base64_data = img.get("base64_data", "")
                if not base64_data:
                    continue
                
                mime_type = img.get("mime_type", "image/png")
                
                # Build data URL
                if base64_data.startswith("data:"):
                    data_url = base64_data
                else:
                    data_url = f"data:{mime_type};base64,{base64_data}"
                
                caption = img.get("caption", "") or img.get("description", "")
                page_num = img.get("page_number")
                
                all_images.append({
                    "data_url": data_url,
                    "caption": caption,
                    "page_number": page_num
                })
        
        if not all_images:
            return ""
        
        # Format as markdown
        markdown = "\n\n---\n\n**📸 Related Images:**\n\n"
        for img in all_images[:5]:  # Limit to 5 images to avoid overwhelming UI
            if img["caption"]:
                markdown += f"*{img['caption']}*\n\n"
            elif img["page_number"]:
                markdown += f"*(Page {img['page_number']})*\n\n"
            
            markdown += f"![Image]({img['data_url']})\n\n"
        
        return markdown
    
    def clear_session(self):
        self.rag_system.reset_thread()
        image_tracker.get_and_clear()