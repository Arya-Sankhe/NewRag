import gradio as gr
from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    demo = create_gradio_ui()
    print("\n🚀 Launching RAG Assistant on http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=custom_css
    )