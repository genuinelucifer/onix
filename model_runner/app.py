#!/usr/bin/env python3
"""
Onix Model Runner — Interactive UI for testing trained models.

Supports:
  - VQ-VAE: Upload image → see reconstruction quality
  - Multimodal: Enter text prompt → generate image
  - LLM: Multi-turn chat conversation

Usage:
    python -m model_runner.app
    python model_runner/app.py
    python model_runner/app.py --port 7861 --device cpu
"""

import os
import sys
from pathlib import Path

# Enable Flash Attention on AMD consumer GPUs before importing PyTorch
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image as PILImage

import gradio as gr
from model_runner.runner import (
    load_model,
    run_vqvae_inference,
    run_multimodal_inference,
    run_llm_inference,
    get_model_info,
    format_model_info,
    LoadedModel,
    GenerationParams,
)


# ---------------------------------------------------------------------------
#  Global state
# ---------------------------------------------------------------------------

_loaded_model: LoadedModel | None = None
_conversation_history: list[dict] = []


# ---------------------------------------------------------------------------
#  Theme + CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Overall polish */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Model info card */
.model-info {
    background: linear-gradient(135deg, #f0fdfa 0%, #e0f2fe 50%, #dbeafe 100%);
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 24px;
    color: #1e293b;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 15px;
    line-height: 1.8;
}
.model-info strong {
    color: #0284c7;
}

/* Status badges */
.status-loaded {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    text-align: center;
}
.status-empty {
    background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
    color: #475569;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
    text-align: center;
}

/* Image output styling */
.generated-image img {
    border-radius: 8px;
    border: 2px solid #cbd5e1;
}

/* Chatbot styling */
.chat-window {
    min-height: 500px;
    font-size: 16px !important;
}

/* Hide progressive loading element for specific containers */
.hide-progress [class*="progress"],
.hide-progress [class*="generating"],
.hide-progress [class*="loading"],
.hide-progress [class*="loader"],
.hide-progress [class*="pending"] {
    display: none !important;
    opacity: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}
"""

_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    text_size=gr.themes.sizes.text_lg,
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_label_text_color="#475569",
    block_title_text_color="#0f172a",
    input_background_fill="#f1f5f9",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    body_text_color="#334155",
)

JS_SHOW_OVERLAY = """
function() {
    let overlay = document.getElementById('custom-loader-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'custom-loader-overlay';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100vw';
        overlay.style.height = '100vh';
        overlay.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        overlay.style.zIndex = '9999';
        overlay.style.display = 'flex';
        overlay.style.justifyContent = 'center';
        overlay.style.alignItems = 'center';
        overlay.style.fontSize = '2rem';
        overlay.style.fontWeight = 'bold';
        overlay.style.color = '#333';
        overlay.style.backdropFilter = 'blur(4px)';
        overlay.innerHTML = 'Loading Model to GPU... Please wait.';
        document.body.appendChild(overlay);
    }
    overlay.style.display = 'flex';
}
"""

JS_HIDE_OVERLAY = """
function() {
    let overlay = document.getElementById('custom-loader-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}
"""



# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def pick_model_path():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(title="Select Model Checkpoint")
        root.destroy()
        return path or ""
    except Exception as e:
        print(f"File dialog error: {e}")
        return ""

def pick_config_path():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(title="Select Config JSON")
        root.destroy()
        return path or ""
    except Exception as e:
        print(f"File dialog error: {e}")
        return ""

def load_model_handler(checkpoint_path: str, device: str, config_path: str, precision: str, compile_model: bool, compile_mode: str):
    """Handle model loading from the UI."""
    global _loaded_model, _conversation_history

    if not checkpoint_path or not checkpoint_path.strip():
        return (
            "<div class='status-empty'>No checkpoint path provided</div>",
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    checkpoint_path = checkpoint_path.strip()
    config_path = config_path.strip() if config_path else None

    # Map precision string to torch.dtype
    dtype = None
    if precision == "bfloat16":
        dtype = torch.bfloat16
    elif precision == "float16":
        dtype = torch.float16
    elif precision == "float32":
        dtype = torch.float32
    elif precision in ("int8", "int4"):
        dtype = precision

    try:
        _loaded_model = load_model(
            checkpoint_path,
            device=device,
            config_path=config_path,
            dtype=dtype,
            compile=compile_model,
            compile_mode=compile_mode,
        )
        _conversation_history = []

        info = get_model_info(_loaded_model)
        info_text = format_model_info(info)
        model_type = _loaded_model.model_type

        status_html = f"<div class='status-loaded'>✅ Model loaded — {model_type.upper()}</div>"

        return (
            status_html,
            gr.update(value=info_text, visible=True),
            gr.update(visible=(model_type == "vqvae")),
            gr.update(visible=(model_type == "multimodal")),
            gr.update(visible=(model_type == "llm")),
        )

    except Exception as e:
        return (
            f"<div class='status-empty'>❌ Failed to load: {e}</div>",
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

def unload_model_handler():
    """Unload model and free GPU memory."""
    global _loaded_model, _conversation_history
    
    _loaded_model = None
    _conversation_history = []
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return (
        "<div class='status-empty'>No model loaded</div>",
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


# ---------------------------------------------------------------------------
#  VQ-VAE handlers
# ---------------------------------------------------------------------------

def vqvae_reconstruct(image):
    """Handle VQ-VAE reconstruction."""
    if _loaded_model is None or _loaded_model.model_type != "vqvae":
        return None, "No VQ-VAE model loaded"

    if image is None:
        return None, "Please upload an image"

    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(image)

    recon, num_tokens, usage = run_vqvae_inference(_loaded_model, image)

    stats = (
        f"**Tokens:** {num_tokens} "
        f"({_loaded_model.vqvae_config.latent_grid_size}×"
        f"{_loaded_model.vqvae_config.latent_grid_size} grid)\n\n"
        f"**Codebook usage:** {usage:.1%} "
        f"({int(usage * _loaded_model.vqvae_config.codebook_size)}"
        f"/{_loaded_model.vqvae_config.codebook_size} entries)"
    )

    return recon, stats


# ---------------------------------------------------------------------------
#  Multimodal handlers
# ---------------------------------------------------------------------------

def multimodal_generate(prompt, temperature, top_k, top_p):
    """Handle text-to-image generation."""
    if _loaded_model is None or _loaded_model.model_type != "multimodal":
        return None, "No multimodal model loaded"

    if not prompt or not prompt.strip():
        return None, "Please enter a text prompt"

    params = GenerationParams(
        temperature=temperature,
        top_k=int(top_k) if top_k else None,
        top_p=top_p if top_p and top_p < 1.0 else None,
    )

    image, status = run_multimodal_inference(_loaded_model, prompt.strip(), params)
    return image, status


# ---------------------------------------------------------------------------
#  LLM chat handlers
# ---------------------------------------------------------------------------

def llm_user_add(message, chat_history):
    """Handle the user message addition: clear input and disable it."""
    if not message or not message.strip():
        # Return same values to avoid error
        return gr.update(), chat_history or []
    
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message.strip()})
    
    return gr.update(value="", interactive=False), chat_history


def make_tps_html(tps: float | None) -> str:
    if tps is None:
        return """
        <div style='background-color: #f8fafc; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;'>
          <div style='font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Decode Speed</div>
          <div style='font-size: 20px; font-weight: bold; color: #334155; margin-top: 4px;'>-- tk/sec</div>
        </div>
        """
    return f"""
    <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #bbf7d0;'>
      <div style='font-size: 11px; color: #166534; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Decode Speed</div>
      <div style='font-size: 20px; font-weight: bold; color: #14532d; margin-top: 4px;'>{tps:.1f} tk/sec</div>
    </div>
    """


def make_ttft_html(ttft_ms: float | None) -> str:
    if ttft_ms is None:
        return """
        <div style='background-color: #f8fafc; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;'>
          <div style='font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Prefill (TTFT)</div>
          <div style='font-size: 20px; font-weight: bold; color: #334155; margin-top: 4px;'>-- ms</div>
        </div>
        """
    return f"""
    <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #bae6fd;'>
      <div style='font-size: 11px; color: #075985; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>Prefill (TTFT)</div>
      <div style='font-size: 20px; font-weight: bold; color: #0c4a6e; margin-top: 4px;'>{ttft_ms:.0f} ms</div>
    </div>
    """


def llm_bot_gen(chat_history, temperature, top_k, top_p,
                max_new_tokens, repetition_penalty, use_kv_cache):
    """Actually run inference and add the bot's response bubble."""
    if _loaded_model is None or _loaded_model.model_type != "llm":
        chat_history = chat_history or []
        chat_history.append({
            "role": "assistant", 
            "content": "⚠️ No LLM model loaded. Please load a model first."
        })
        return chat_history, gr.update(interactive=True), make_tps_html(None), make_ttft_html(None)

    if not chat_history:
        return chat_history, gr.update(interactive=True), make_tps_html(None), make_ttft_html(None)

    # Build conversation text
    conv_text = ""
    for msg in chat_history:
        # Robust access for both dicts and ChatMessage objects
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
        
        # Handle cases where content is a list (multimodal/segment format)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            content = "".join(parts)
            
        conv_text += str(content)

    params = GenerationParams(
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        top_k=int(top_k) if top_k else None,
        top_p=top_p if top_p and top_p < 1.0 else None,
        repetition_penalty=repetition_penalty,
        use_kv_cache=use_kv_cache,
    )

    tps_val, ttft_val = None, None
    try:
        res = run_llm_inference(_loaded_model, conv_text, params)
        response = res["text"]
        tokens = res["tokens"]
        ttft = res["ttft"]
        decode_time = res["decode_time"]
        
        decode_tokens = max(0, tokens - 1)
        tps_val = decode_tokens / decode_time if decode_time > 0 else 0.0
        ttft_val = ttft * 1000
        chat_history.append({"role": "assistant", "content": response})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ Generation error: {e}"})

    return chat_history, gr.update(interactive=True), make_tps_html(tps_val), make_ttft_html(ttft_val)


def clear_chat():
    """Clear chat history."""
    return [], "", make_tps_html(None), make_ttft_html(None)


# ---------------------------------------------------------------------------
#  Build the Gradio UI
# ---------------------------------------------------------------------------

def create_ui():
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(
        title="Onix Model Runner",
    ) as app:

        # ---- Header ----
        gr.Markdown(
            """
            # 🧪 Onix Model Runner
            Load and test your trained models — VQ-VAE reconstruction,
            text-to-image generation, or LLM chat.
            """,
        )

        # ---- Model Loading Section ----
        with gr.Group():
            gr.Markdown("### 📂 Load Model")
            with gr.Row():
                checkpoint_input = gr.Textbox(
                    label="Checkpoint path",
                    placeholder="models/my-model/checkpoint_final.pt  (or just the model directory)",
                    scale=10,
                    lines=2,
                    interactive=True,
                )
                browse_model_btn = gr.Button("📁", scale=1, min_width=50)

            with gr.Row():
                config_input = gr.Textbox(
                    label="Config path (Optional)",
                    placeholder="Path to config.json if not alongside model",
                    scale=10,
                    lines=2,
                    interactive=True,
                )
                browse_config_btn = gr.Button("📁", scale=1, min_width=50)
                
            with gr.Row():
                device_dropdown = gr.Dropdown(
                    label="Device",
                    choices=["cuda", "cpu"] + [f"cuda:{i}" for i in range(8)],
                    value="cuda",
                    interactive=True,
                    scale=2,
                )
                precision_dropdown = gr.Dropdown(
                    label="Precision Mode",
                    choices=["float32", "bfloat16", "float16", "int8", "int4"],
                    value="float16",
                    interactive=True,
                    scale=2,
                )
                compile_checkbox = gr.Checkbox(
                    label="Compile Model",
                    value=False,
                    interactive=True,
                    scale=1,
                )
                compile_mode_dropdown = gr.Dropdown(
                    label="Compile Mode",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    value="default",
                    interactive=True,
                    scale=2,
                )
                load_btn = gr.Button("🔄 Load", variant="primary", scale=1)
                unload_btn = gr.Button("🗑️ Unload", variant="stop", scale=1)

            model_status = gr.HTML(
                value="<div class='status-empty'>No model loaded</div>",
            )
            model_info_html = gr.HTML(
                visible=False,
                elem_classes=["model-info"],
            )

        # ---- VQ-VAE Tab ----
        with gr.Group(visible=False) as vqvae_section:
            gr.Markdown("### 🖼️ VQ-VAE — Image Reconstruction")
            gr.Markdown(
                "*Upload an image to see how well the VQ-VAE compresses and "
                "reconstructs it.*"
            )
            with gr.Row(equal_height=True):
                with gr.Column():
                    vqvae_input = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=320,
                    )
                    vqvae_btn = gr.Button(
                        "🔄 Reconstruct", variant="primary"
                    )
                with gr.Column():
                    vqvae_output = gr.Image(
                        label="Reconstruction",
                        type="pil",
                        height=320,
                        interactive=False,
                        elem_classes=["generated-image"],
                    )
                    vqvae_stats = gr.Markdown("", elem_classes=["hide-progress"])

        # ---- Multimodal Tab ----
        with gr.Group(visible=False) as multimodal_section:
            gr.Markdown("### 🎨 Multimodal — Text to Image")
            gr.Markdown(
                "*Enter a text prompt to generate a pixel art image.*"
            )
            with gr.Row(equal_height=True):
                with gr.Column():
                    mm_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="A colorful pixel art character with a sword",
                        lines=3,
                    )
                    with gr.Row():
                        mm_temp = gr.Slider(
                            0.1, 2.0, value=0.9, step=0.05,
                            label="Temperature",
                        )
                        mm_topk = gr.Slider(
                            1, 500, value=100, step=1,
                            label="Top-K",
                        )
                        mm_topp = gr.Slider(
                            0.0, 1.0, value=0.95, step=0.05,
                            label="Top-P",
                        )
                    mm_btn = gr.Button(
                        "🎨 Generate Image", variant="primary"
                    )
                with gr.Column():
                    mm_output = gr.Image(
                        label="Generated Image",
                        type="pil",
                        height=320,
                        interactive=False,
                        elem_classes=["generated-image"],
                    )
                    mm_status = gr.Markdown("", elem_classes=["hide-progress"])

        # ---- LLM Chat Tab ----
        with gr.Group(visible=False) as llm_section:
            gr.Markdown("### 💬 LLM — Text Generation Chat")
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=450,
                        elem_classes=["chat-window"],
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            scale=5,
                            lines=1,
                            elem_classes=["hide-progress"],
                        )
                        chat_send = gr.Button(
                            "Send", variant="primary", scale=1
                        )
                    with gr.Row():
                        chat_clear = gr.Button("🗑️ Clear Chat", size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("**Generation Settings**")
                    llm_use_kv_cache = gr.Checkbox(
                        value=True,
                        label="Use KV Cache",
                    )
                    with gr.Accordion("Text Generation Parameters", open=False):
                        llm_temp = gr.Slider(
                            0.0, 2.0, value=0.8, step=0.05,
                            label="Temperature",
                        )
                        llm_topk = gr.Slider(
                            0, 500, value=50, step=1,
                            label="Top-K (0=off)",
                        )
                        llm_topp = gr.Slider(
                            0.0, 1.0, value=0.9, step=0.05,
                            label="Top-P",
                        )
                        llm_max_tokens = gr.Slider(
                            10, 2000, value=200, step=10,
                            label="Max New Tokens",
                        )
                        llm_rep_penalty = gr.Slider(
                            1.0, 2.0, value=1.1, step=0.05,
                            label="Repetition Penalty",
                        )
                    gr.Markdown("---")
                    gr.Markdown("**Performance Metrics**")
                    with gr.Row():
                        llm_tps_output = gr.HTML(value=make_tps_html(None))
                        llm_ttft_output = gr.HTML(value=make_ttft_html(None))

        # ---- Event bindings ----

        # Browse buttons
        browse_model_btn.click(
            fn=pick_model_path,
            outputs=[checkpoint_input],
            show_progress="hidden",
        )
        browse_config_btn.click(
            fn=pick_config_path,
            outputs=[config_input],
            show_progress="hidden",
        )

        # Load model
        load_btn.click(
            fn=lambda: None, js=JS_SHOW_OVERLAY
        ).then(
            fn=load_model_handler,
            inputs=[checkpoint_input, device_dropdown, config_input, precision_dropdown, compile_checkbox, compile_mode_dropdown],
            outputs=[model_status, model_info_html, vqvae_section, multimodal_section, llm_section],
            show_progress="hidden",
        ).then(
            fn=lambda: None, js=JS_HIDE_OVERLAY
        )
        
        # Also load on enter key in path input
        checkpoint_input.submit(
            fn=lambda: None, js=JS_SHOW_OVERLAY
        ).then(
            fn=load_model_handler,
            inputs=[checkpoint_input, device_dropdown, config_input, precision_dropdown, compile_checkbox, compile_mode_dropdown],
            outputs=[model_status, model_info_html, vqvae_section, multimodal_section, llm_section],
            show_progress="hidden",
        ).then(
            fn=lambda: None, js=JS_HIDE_OVERLAY
        )
        
        config_input.submit(
            fn=lambda: None, js=JS_SHOW_OVERLAY
        ).then(
            fn=load_model_handler,
            inputs=[checkpoint_input, device_dropdown, config_input, precision_dropdown, compile_checkbox, compile_mode_dropdown],
            outputs=[model_status, model_info_html, vqvae_section, multimodal_section, llm_section],
            show_progress="hidden",
        ).then(
            fn=lambda: None, js=JS_HIDE_OVERLAY
        )
        
        # Unload model
        unload_btn.click(
            fn=unload_model_handler,
            outputs=[model_status, model_info_html, vqvae_section, multimodal_section, llm_section],
            show_progress="hidden",
        )

        # VQ-VAE
        vqvae_btn.click(
            fn=vqvae_reconstruct,
            inputs=[vqvae_input],
            outputs=[vqvae_output, vqvae_stats],
        )

        # Multimodal
        mm_btn.click(
            fn=multimodal_generate,
            inputs=[mm_prompt, mm_temp, mm_topk, mm_topp],
            outputs=[mm_output, mm_status],
        )

        # LLM Chat
        chat_inputs = [
            chat_input, chatbot,
            llm_temp, llm_topk, llm_topp,
            llm_max_tokens, llm_rep_penalty,
        ]
        
        # User step: Add msg, clear + disable input
        chat_send.click(
            fn=llm_user_add,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot],
            show_progress="minimal",
        ).then(
            fn=llm_bot_gen,
            inputs=[chatbot, llm_temp, llm_topk, llm_topp, llm_max_tokens, llm_rep_penalty, llm_use_kv_cache],
            outputs=[chatbot, chat_input, llm_tps_output, llm_ttft_output],
            show_progress="minimal",
        )
        chat_input.submit(
            fn=llm_user_add,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot],
            show_progress="minimal",
        ).then(
            fn=llm_bot_gen,
            inputs=[chatbot, llm_temp, llm_topk, llm_topp, llm_max_tokens, llm_rep_penalty, llm_use_kv_cache],
            outputs=[chatbot, chat_input, llm_tps_output, llm_ttft_output],
            show_progress="minimal",
        )
        
        chat_clear.click(
            fn=clear_chat,
            outputs=[chatbot, chat_input, llm_tps_output, llm_ttft_output],
            show_progress="minimal",
        )

        chatbot.clear(
            fn=clear_chat,
            outputs=[chatbot, chat_input, llm_tps_output, llm_ttft_output],
            show_progress="minimal",
        )

    return app


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Onix Model Runner")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--device", type=str, default=None,
                        help="Override default device in the dropdown")
    args = parser.parse_args()

    app = create_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
