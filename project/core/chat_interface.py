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
            Markdown string with relevant images, or empty string
        """
        # Collect all images from retrieved parents
        all_images = []
        
        for parent_id in parent_ids:
            parent_data = self.parent_store.load(parent_id)
            if not parent_data:
                continue
            
            ocr_images = parent_data.get("metadata", {}).get("ocr_images", [])
            for img in ocr_images:
                # Only include images with base64 data
                if img.get("base64_data"):
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
        
        # Format as markdown
        return self._format_images_markdown(relevant_images)
    
    def _format_images_markdown(self, images: list) -> str:
        """Format scored images as inline HTML with click-to-enlarge functionality."""
        
        # Use HTML for clickable images with lightbox-style enlargement
        html = '\n\n<div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #444;">'
        html += '<p style="margin-bottom: 10px; font-weight: bold;">📸 Related Images:</p>'
        html += '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
        
        for i, img in enumerate(images):
            base64_data = img.get("base64_data", "")
            mime_type = img.get("mime_type", "image/png")
            
            # Build data URL
            if base64_data.startswith("data:"):
                data_url = base64_data
            else:
                data_url = f"data:{mime_type};base64,{base64_data}"
            
            # Caption and score
            caption = img.get("caption", "") or img.get("description", "")
            score = img.get("relevance_score", 0)
            page_num = img.get("page_number")
            
            # Build caption text
            if caption:
                caption_text = f"{caption} ({score:.0%})"
            elif page_num:
                caption_text = f"Page {page_num} ({score:.0%})"
            else:
                caption_text = f"Image ({score:.0%})"
            
            # Create clickable thumbnail with CSS for enlargement
            # Uses a link that opens in new tab for enlargement (Gradio-compatible)
            html += f'''
            <div style="flex: 0 0 auto; text-align: center;">
                <a href="{data_url}" target="_blank" title="Click to enlarge: {caption_text}">
                    <img src="{data_url}" 
                         alt="{caption_text}"
                         style="max-width: 200px; max-height: 150px; border-radius: 8px; 
                                cursor: zoom-in; border: 1px solid #555;
                                transition: transform 0.2s ease, box-shadow 0.2s ease;"
                         onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.3)';"
                         onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';"
                    />
                </a>
                <p style="font-size: 11px; color: #aaa; margin-top: 5px; max-width: 200px;">{caption_text}</p>
            </div>
            '''
        
        html += '</div></div>'
        
        return html
    
    def clear_session(self):
        self.rag_system.reset_thread()
        image_tracker.get_and_clear()