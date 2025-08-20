import argparse
from typing import List, Optional, Tuple

import PIL.Image
import gradio as gr
import moviepy.editor as mp
import numpy as np
import torch
from ovis.model.modeling_ovis import Ovis 

model: Ovis = None

def load_video_frames(video_path: Optional[str], n_frames: int = 8) -> Optional[List[PIL.Image.Image]]:
    """Extract a fixed number of frames from the video file."""
    if not video_path:
        return None
    try:
        with mp.VideoFileClip(video_path) as clip:
            duration = clip.duration
            if duration is None or clip.fps is None or duration <= 0 or clip.fps <= 0:
                print(f"Warning: Unable to process video {video_path}. Invalid duration or fps.")
                return None
            
            total_possible_frames = int(duration * clip.fps)
            num_to_extract = min(n_frames, total_possible_frames)

            if num_to_extract <= 0:
                print(f"Warning: Cannot extract frames from {video_path}. Computed extractable frames is zero.")
                return None
            
            frames = []
            timestamps = np.linspace(0, duration, num_to_extract, endpoint=True)
            for t in timestamps:
                frame_np = clip.get_frame(t)
                frames.append(PIL.Image.fromarray(frame_np))
        print(f"Successfully extracted {len(frames)} frames from {video_path}.")
        return frames
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def run_single_model(
    image_input: Optional[PIL.Image.Image], 
    video_input: Optional[str], 
    prompt: str,
    do_sample: bool, 
    max_new_tokens: int, 
    enable_thinking: bool
) -> str:
    """Run single model inference."""
    if not prompt and not image_input and not video_input:
        gr.Warning("Please enter a prompt, upload an image, or upload a video.")
        return ""

    # Prepare vision inputs
    images = [image_input] if image_input else None
    video_frames = load_video_frames(video_input)
    videos = [video_frames] if video_frames else None
    
    # Construct full prompt with placeholders
    visual_placeholders = ('<image>\n' * len(images) if images else "") + ('<video>\n' if videos else "")
    full_prompt = visual_placeholders + prompt
    
    # Call model chat method
    response, thinking, _ = model.chat(
        prompt=full_prompt, 
        history=None,  # Always start a new conversation
        images=images, 
        videos=videos,
        do_sample=do_sample, 
        max_new_tokens=max_new_tokens, 
        enable_thinking=enable_thinking,
    )
    
    # Format output
    if enable_thinking and thinking:
        return f"**Thinking:**\n```text\n{thinking}\n```\n\n**Response:**\n{response}"
    return response

def toggle_media_input(choice: str) -> Tuple[gr.update, gr.update]:
    """Toggle visibility of image and video input components."""
    if choice == "Image":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
    else:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)

def clear_interface() -> Tuple[str, None, None, str, str]:
    """Reset all inputs and outputs."""
    return "", None, None, "", "Image"

def start_generation() -> Tuple[gr.update, gr.update, gr.update]:
    """Update UI status when generation starts."""
    return (
        gr.update(value="⏳ Generating...", interactive=False),
        gr.update(interactive=False),
        gr.update(value="⏳ Model is generating...")
    )

def finish_generation() -> Tuple[gr.update, gr.update]:
    """Restore UI status after generation ends."""
    return gr.update(value="Generate", interactive=True), gr.update(interactive=True)

def build_demo(model_path: str, gpu: int):
    """Build single-model Gradio demo interface."""
    global model
    device = f"cuda:{gpu}"
    print(f"Loading model {model_path} to device {device}...")
    model = Ovis.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device).eval()
    print("Model loaded successfully.")

    custom_css = "#output_md .prose { font-size: 18px !important; }"
    with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as demo:
        gr.Markdown("# Multimodal Large Language Model Interface")
        gr.Markdown(f"Running on **GPU {gpu}**. Each submission starts a new conversation.")
        
        with gr.Row():
            # Left column - inputs
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                input_type_radio = gr.Radio(
                    choices=["Image", "Video"], value="Image", label="Select Input Type"
                )
                image_input = gr.Image(label="Image Input", type="pil", visible=True, height=400)
                video_input = gr.Video(label="Video Input", visible=False)
                prompt_input = gr.Textbox(
                    label="Prompt", placeholder="Enter your prompt here... (Press Enter to submit)", lines=3
                )
                with gr.Accordion("Generation Settings", open=True):
                    do_sample = gr.Checkbox(label="Enable Sampling (Do Sample)", value=False)
                    max_new_tokens = gr.Slider(
                        minimum=32, maximum=2048, value=1024, step=32, label="Max New Tokens"
                    )
                    enable_thinking = gr.Checkbox(label="Deep Thinking", value=False)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    generate_btn = gr.Button("Generate", variant="primary", scale=2)

            # Right column - output
            with gr.Column(scale=2):
                model_name = model_path.split('/')[-1]
                gr.Markdown(f"### Model Output\n`{model_name}`")
                output_display = gr.Markdown(elem_id="output_md")
        
        # Event handlers
        input_type_radio.change(
            fn=toggle_media_input, 
            inputs=input_type_radio, 
            outputs=[image_input, video_input]
        )
        
        run_inputs = [image_input, video_input, prompt_input, do_sample, max_new_tokens, enable_thinking]

        generate_btn.click(
            fn=start_generation, 
            outputs=[generate_btn, clear_btn, output_display]
        ).then(
            fn=run_single_model,
            inputs=run_inputs,
            outputs=[output_display]
        ).then(
            fn=finish_generation,
            outputs=[generate_btn, clear_btn]
        )

        prompt_input.submit(
            fn=start_generation, 
            outputs=[generate_btn, clear_btn, output_display]
        ).then(
            fn=run_single_model,
            inputs=run_inputs,
            outputs=[output_display]
        ).then(
            fn=finish_generation,
            outputs=[generate_btn, clear_btn]
        )
        
        clear_btn.click(
            fn=clear_interface,
            outputs=[output_display, image_input, video_input, prompt_input, input_type_radio]
        ).then(
            fn=toggle_media_input, 
            inputs=input_type_radio, 
            outputs=[image_input, video_input]
        )
        
    return demo

def parse_args():
    parser = argparse.ArgumentParser(description="Gradio interface for Ovis.")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to run the model.")
    parser.add_argument("--port", type=int, default=9901, help="Port to run the Gradio service.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name for Gradio app.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    demo = build_demo(
        model_path=args.model_path,
        gpu=args.gpu
    )
    
    print(f"Launching Gradio app at http://{args.server_name}:{args.port}")
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.port,
        share=False,
        ssl_verify=False
    )
