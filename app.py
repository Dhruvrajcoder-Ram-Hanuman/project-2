import os
import uvicorn
import gradio as gr
from server import app as fastapi_app

# Gradio demo block
with gr.Blocks(title="SENTINEL ANPR - Vehicle Intelligence Command Center") as demo:
    pass

# Mount Gradio onto our FastAPI application
# Root / serves our complete dashboard with all 4 cameras, map, routes, and scanner!
# /gradio serves the Gradio interface
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[*] Starting SENTINEL ANPR on Hugging Face Spaces (Port: {port})")
    uvicorn.run(app, host="0.0.0.0", port=port)
