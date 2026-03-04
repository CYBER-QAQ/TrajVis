import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from video_trajectory_visualizer import visualize_trajectory, visualize_trajectory_absolute

import re
import subprocess

def convert_video_for_web(video_path):
    if not video_path:
        return None
    print(f"🔄 Automatically transcoding video for browser playback: {video_path}")
    # Generate a new transcoded file path
    out_path = video_path.rsplit('.', 1)[0] + "_web.mp4"
    
    # Force conversion to standard H.264 + AAC (fast preset to speed up conversion)
    cmd = [
        'ffmpeg', '-y', '-i', video_path, 
        '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', 
        '-pix_fmt', 'yuv420p', '-acodec', 'aac', out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Transcoding successful: {out_path}")
        return out_path
    except Exception as e:
        print(f"❌ Transcoding failed, trying original video: {e}")
        return video_path

def hex_to_rgb(hex_color):
    if not isinstance(hex_color, str):
        print(f"⚠️ Color parsing failed: input is not a string, original value {repr(hex_color)}")
        return (255, 150, 0)
        
    hex_color = hex_color.strip()
    
    # Try to match rgb(r, g, b) or rgba(r, g, b, a) formats, supporting decimals
    rgb_match = re.match(r'rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)', hex_color, re.IGNORECASE)
    if rgb_match:
        return (int(float(rgb_match.group(1))), int(float(rgb_match.group(2))), int(float(rgb_match.group(3))))
        
    hex_core = hex_color.lstrip('#')
    
    # Compatible with Gradio's 8-bit Hex with transparency or shorthand 3-bit Hex
    if len(hex_core) == 8:
        hex_core = hex_core[:6]  # Discard Alpha channel
    elif len(hex_core) == 3:
        hex_core = ''.join([c*2 for c in hex_core]) # #abc -> #aabbcc
        
    if len(hex_core) != 6:
        print(f"⚠️ Color parsing failed: unknown length or invalid format '{hex_color}', fallback to orange")
        return (255, 150, 0)
        
    try:
        r = int(hex_core[0:2], 16)
        g = int(hex_core[2:4], 16)
        b = int(hex_core[4:6], 16)
        return (r, g, b)
    except ValueError as e:
        print(f"⚠️ Hex color parsing failed: {str(e)} original value '{hex_color}'")
        return (255, 150, 0)

def process_video(video_path, n_frames, blend_mode, trim_start, trim_end, color_hex):
    print(f"\n--- 🚀 New Render Request ---")
    print(f"Received raw color_hex from Gradio: {repr(color_hex)}")
    
    if not video_path:
        return None, None, "⚠️ Error: Please upload or select a video file first", gr.update(choices=[], value=None), None

    out_jpg = "trajectory_result.jpg"
    out_pdf = "trajectory_result.pdf"
    
    # Parse start and end times from sliders
    trim_start = float(trim_start) if trim_start is not None else 0.0
    trim_end = float(trim_end) if trim_end is not None else 0.0
        
    color_rgb = hex_to_rgb(color_hex)
    print(f"🎨 RGB result after hex_to_rgb parsing: {color_rgb}")

    try:
        # Modify parameter logic. Originally trim_end meant 'duration to skip at the end',
        # But here time_range is 'absolute video end time', so we use absolute timestamp directly instead of subtracting from length
        res_jpg, res_pdf, layers = visualize_trajectory_absolute(
            video_path, out_jpg, out_pdf, int(n_frames), blend_mode, 
            trim_start, trim_end, color_rgb
        )
        
        if not res_jpg or not os.path.exists(res_jpg):
            return None, None, "❌ Video duration too short or generation failed.", gr.update(choices=[], value=None), None
            
        # load composite image
        img_bgr = cv2.imread(res_jpg)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        editor_val = {"background": img_rgb, "layers": [], "composite": img_rgb}
        
        if layers:
            choices = list(layers.keys())
            return editor_val, [out_jpg, out_pdf], "✅ Trajectory generated successfully! Select a brush layer on the bottom left, then paint on the canvas to reveal that layer!", gr.update(choices=choices, value=choices[0]), layers
        else:
            return editor_val, [out_jpg, out_pdf], "✅ Generation successful! Current mode does not support brush fine-tuning.", gr.update(choices=[], value=None), None

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", gr.update(choices=[], value=None), None

LAYER_COLOR_MAP = {
    'First Frame (Original)': "#ff0000",
    'Last Frame (Original)': "#00ff00",
    'Middle Trajectory Frame': "#0000ff",
    'Clean Pure Background': "#ffff00",
    'Composite Image (All Trajectory)': "#00ffff",
    'Middle Frame': "#0000ff",
    'Background': "#ffff00"
}

def hex_to_rgb_tuple(h):
    h = h.strip()
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def create_feathered_mask(drawn_layers, shape, feather_val, target_hex):
    """Generate core mask based on strokes, only processing target color strokes (based on LAYER_COLOR_MAP), enabling parallel undo and isolation of multiple brush types"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    target_r, target_g, target_b = hex_to_rgb_tuple(target_hex)
    
    for stroke in drawn_layers:
        if stroke.ndim == 3 and stroke.shape[-1] == 4:
            alpha = stroke[:, :, 3]
            r = stroke[:, :, 0].astype(np.int16)
            g = stroke[:, :, 1].astype(np.int16)
            b = stroke[:, :, 2].astype(np.int16)
            
            # Use color distance to separate brush tolerance
            dist = np.abs(r - target_r) + np.abs(g - target_g) + np.abs(b - target_b)
            mask[(alpha > 10) & (dist < 100)] = 255
            
    if feather_val <= 1 or np.max(mask) == 0:
        return mask.astype(np.float32) / 255.0

    mask_smooth = cv2.GaussianBlur(mask, (7, 7), 0)
    _, binary_mask = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
    
    if np.max(binary_mask) == 0:
        return mask.astype(np.float32) / 255.0
    
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    alpha_map = dist_transform / max(feather_val, 1)
    alpha_map = np.clip(alpha_map, 0.0, 1.0)
    
    blur_kernel = int(feather_val) // 2
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel < 3:
        blur_kernel = 3
        
    alpha_map = cv2.GaussianBlur(alpha_map, (blur_kernel, blur_kernel), 0)
    
    return alpha_map

def apply_all_layers(base_img, drawn_layers, layers, feather_val):
    new_base_float = base_img.copy().astype(np.float32)
    # Apply sequentially
    for layer_name, hex_color in LAYER_COLOR_MAP.items():
        if layer_name in layers:
            replacement = layers[layer_name]
            alpha_map = create_feathered_mask(drawn_layers, base_img.shape, feather_val, hex_color)
            if np.max(alpha_map) > 0:
                alpha_map_3d = np.expand_dims(alpha_map, axis=-1)
                
                # Adapt channels
                if replacement.shape[-1] == 4 and new_base_float.shape[-1] == 3:
                    replacement = replacement[:, :, :3]
                elif replacement.shape[-1] == 3 and new_base_float.shape[-1] == 4:
                    new_base_float = new_base_float[:, :, :3]
                    
                replacement_float = replacement.astype(np.float32)
                # Overlay and blend sequentially
                new_base_float = new_base_float * (1.0 - alpha_map_3d) + replacement_float * alpha_map_3d

    return np.clip(new_base_float, 0, 255).astype(np.uint8)

def apply_brush(editor_dict, selected_layer, layers, feather_val):
    if layers is None or not editor_dict:
        return editor_dict, gr.update()
        
    base_img = editor_dict.get("background")
    drawn_layers = editor_dict.get("layers", [])
    
    if base_img is None or not drawn_layers:
        return editor_dict, gr.update()
        
    new_base = apply_all_layers(base_img, drawn_layers, layers, feather_val)
    
    save_img = new_base[:, :, :3] if new_base.shape[-1] == 4 else new_base
    new_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("trajectory_result.jpg", new_bgr)
    img = Image.fromarray(save_img)
    img.save("trajectory_result.pdf", 'PDF', resolution=100.0)
    
    return {
        "background": new_base,
        "layers": [],  
        "composite": new_base
    }, ["trajectory_result.jpg", "trajectory_result.pdf"]

def live_preview_brush(editor_dict, selected_layer, layers, feather_val):
    """Real-time preview during painting without hard-writing to layers"""
    if layers is None or not editor_dict:
        return None
        
    base_img = editor_dict.get("background")
    drawn_layers = editor_dict.get("layers", [])
    
    if base_img is None:
        return None
    if not drawn_layers:
        return base_img
        
    return apply_all_layers(base_img, drawn_layers, layers, feather_val)

def update_time_slider(video_path):
    """Read video duration and update slider range after user uploads a video"""
    if not video_path:
        return gr.update(minimum=0, maximum=10, value=0, interactive=False), gr.update(minimum=0, maximum=10, value=10, interactive=False)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return gr.update(), gr.update()
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0:
        return gr.update(), gr.update()
        
    duration = total_frames / fps
    return gr.update(
        minimum=0.0, 
        maximum=duration, 
        value=0.0, 
        interactive=True,
        label=f"⏱️ Trim Start Time (s) (Total: {duration:.2f} s)"
    ), gr.update(
        minimum=0.0, 
        maximum=duration, 
        value=duration, 
        interactive=True,
        label=f"⏱️ Trim End Time (s) (Total: {duration:.2f} s)"
    )

with gr.Blocks(title="Video Motion Trajectory Generator") as demo:
    state_layers = gr.State(None)
    
    gr.Markdown("# 🏃‍♂️ Intelligent Video Motion Trajectory Visualizer")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📂 Step 1: Drag & Drop Video")
            
            gr.Markdown("### ⚙️ Step 2: Adjust Parameters")
            n_slider = gr.Slider(minimum=2, maximum=150, value=15, step=1, label="Uniformly Sampled Frames (N)")
            
            # Use two separate sliders to control start and end times
            with gr.Row():
                time_trim_start = gr.Slider(
                    minimum=0, maximum=10, value=0, 
                    step=0.1, interactive=False, 
                    label="⏱️ Trim Start Time (Upload video first to read duration)"
                )
                time_trim_end = gr.Slider(
                    minimum=0, maximum=10, value=10, 
                    step=0.1, interactive=False, 
                    label="⏱️ Trim End Time (Upload video first to read duration)"
                )
                
            blend_drop = gr.Dropdown(
                choices=["focus_endpoints", "colored_contours", "smart", "average", "max", "min"],
                value="focus_endpoints",
                label="🎨 Visual Blend Mode"
            )
            color_picker = gr.ColorPicker(value="#00fa96", label="🎭 Trajectory Highlight Color")
            
            submit_btn = gr.Button("🚀 Render Trajectory", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 🖌️ Step 3: Brush Layer Modification (Available after generation)\nPaint on the [Canvas] to erase or reveal content. Drag the slider to see edge feathering!")
            
            with gr.Row():
                brush_type = gr.Dropdown(choices=[], label="🎨 Target Reveal Layer for Brush (Brush changes color automatically to indicate category)")
                # The feather slider now represents the 'actual pixel distance for edge attenuation inward/outward'
                feather_slider = gr.Slider(minimum=0, maximum=100, step=1, value=20, label="🪶 Brush Edge Feather Width (Pixel distance)")
            
            with gr.Row():
                layer_preview = gr.Image(label="🔍 Reference Background to Reveal", interactive=False, visible=False)
                live_result_preview = gr.Image(label="✨ Real-time Feathered Composite Preview (Auto-updates with painting/slider)", interactive=False)
            
            # Brushes come with transparency, giving canvas painting a 'perspective' feel of a mask, rather than stiff solid colors
            image_preview = gr.ImageEditor(
                label="🌟 Canvas (Supports freehand painting)", 
                type="numpy", 
                interactive=True,
                brush=gr.Brush(colors=["#ffffff80"], default_size=30)
            )
            
            with gr.Row():
                apply_btn = gr.Button("✅ Confirm & Merge Feathered Painting to Final Result", variant="primary")
                files_download = gr.File(label="📦 Download Composite Image and PDF Files")

    submit_btn.click(
        fn=process_video,
        inputs=[video_input, n_slider, blend_drop, time_trim_start, time_trim_end, color_picker],
        outputs=[image_preview, files_download, status_text, brush_type, state_layers]
    )
    
    # Bind real-time preview update events for painting, brush switching, and feather value changes
    for input_comp in [image_preview, brush_type, feather_slider]:
        input_comp.change(
            fn=live_preview_brush,
            inputs=[image_preview, brush_type, state_layers, feather_slider],
            outputs=[live_result_preview]
        )
    
    apply_btn.click(
        fn=apply_brush,
        inputs=[image_preview, brush_type, state_layers, feather_slider],
        outputs=[image_preview, files_download]
    )

    def update_layer_preview_and_brush(selected_layer, layers):
        out_vals = []
        if layers and selected_layer in layers:
            out_vals.append(gr.update(value=layers[selected_layer], visible=True))
        else:
            out_vals.append(gr.update(visible=False))
            
        color_hex = LAYER_COLOR_MAP.get(selected_layer, "#ffffff") + "80" # Add 50% transparency
        out_vals.append(gr.update(brush=gr.Brush(colors=[color_hex], default_size=30)))
        return out_vals

    brush_type.change(
        fn=update_layer_preview_and_brush,
        inputs=[brush_type, state_layers],
        outputs=[layer_preview, image_preview]
    )

    # Read video duration and update timeline slider maximum after upload
    video_input.change(
        fn=update_time_slider,
        inputs=[video_input],
        outputs=[time_trim_start, time_trim_end]
    )

    # Automatically trigger forced transcoding after upload, returning web-compatible video to the component
    video_input.upload(
        fn=convert_video_for_web,
        inputs=[video_input],
        outputs=[video_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, theme=gr.themes.Soft())
