import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem

def create_gradio_ui():
    rag_system = RAGSystem()
    rag_system.initialize()
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return "📭 No documents available in the knowledge base"
        return "\n".join([f"{f}" for f in files])
    
    def upload_handler(files, vlm_choice, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
        
        # Convert Radio choice to boolean
        enable_vlm = (vlm_choice == "Enabled")
            
        added, skipped = doc_manager.add_documents(
            files, 
            enable_vlm=enable_vlm,
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        vlm_status = "with VLM captions" if enable_vlm else "without VLM"
        gr.Info(f"✅ Added: {added} | Skipped: {skipped} ({vlm_status})")
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info(f"🗑️ Removed all documents")
        return format_file_list()
    
    def extract_images_from_state(rag_system) -> list:
        """
        Extract images from the latest agent state.
        
        Looks through the agent's memory for retrieved parent chunks
        that contain images.
        """
        try:
            # Get the current thread state
            config = rag_system.get_config()
            state = rag_system.agent_graph.get_state(config)
            
            if not state or not state.values:
                return []
            
            # Look for images in agent answers or messages
            images = []
            messages = state.values.get("messages", [])
            
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    # Check if message contains image references
                    # Images are embedded in tool call results
                    pass
                    
            return images
        except Exception as e:
            print(f"Could not extract images: {e}")
            return []
    
    def format_response_with_images(response: str, images: list) -> str:
        """
        Append images to response in markdown format.
        
        Args:
            response: The text response from the agent
            images: List of image dicts with data_url, caption, page_number
            
        Returns:
            Response with images appended in markdown format
        """
        if not images:
            return response
        
        image_section = "\n\n---\n\n**📸 Related Images:**\n\n"
        for img in images:
            caption = img.get("caption", "")
            page_num = img.get("page_number")
            data_url = img.get("data_url", "")
            
            if not data_url:
                continue
            
            # Build image markdown
            if caption:
                image_section += f"*{caption}*\n\n"
            elif page_num:
                image_section += f"*(Page {page_num})*\n\n"
            
            image_section += f"![Image]({data_url})\n\n"
        
        return response + image_section
    
    def chat_handler(msg, hist):
        response = chat_interface.chat(msg, hist)
        
        # Try to extract any images from the retrieval
        # Note: Images are currently embedded in parent chunks
        # and passed through tool calls. Future enhancement could
        # parse tool results for image data.
        
        return response
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    with gr.Blocks(title="Agentic RAG") as demo:
        
        with gr.Tab("Documents", elem_id="doc-management-tab"):
            gr.Markdown("## Add New Documents")
            gr.Markdown("Upload PDF or Markdown files. PDFs are processed with OCR and image extraction.")
            
            files_input = gr.File(
                label="Drop PDF or Markdown files here",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False
            )
            
            # VLM Toggle for enhanced image captions - using Radio for better click handling
            vlm_toggle = gr.Radio(
                choices=["Disabled", "Enabled"],
                value="Disabled",
                label="🧠 VLM Captions",
                info="Use AI vision model for detailed image descriptions (adds ~1-2s per image, uses OpenAI API)",
                interactive=True,
                elem_id="vlm-toggle"
            )
            
            add_btn = gr.Button("Add Documents", variant="primary", size="md")
            
            gr.Markdown("## Current Documents in the Knowledge Base")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines = 7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="md")
                clear_btn = gr.Button("Clear All", variant="stop", size="md")
            
            add_btn.click(
                upload_handler, 
                [files_input, vlm_toggle], 
                [files_input, file_list], 
                show_progress="corner"
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=600, 
                placeholder="Ask me anything about your documents! Images from PDFs will be displayed when relevant.",
                show_label=False
            )
            chatbot.clear(clear_chat_handler)
            
            gr.ChatInterface(fn=chat_handler, chatbot=chatbot)
    
    return demo