import cv2
import numpy as np
import argparse
import os
from PIL import Image

def visualize_trajectory(video_path, output_jpg, output_pdf, N=10, blend_mode='colored_contours', trim_start=0.0, trim_end=0.0, mask_color=(255, 150, 0)):
    # Compatible with the original tail truncation logic
    return visualize_trajectory_absolute_core(video_path, output_jpg, output_pdf, N, blend_mode, trim_start, trim_end, mask_color, is_absolute=False)

def visualize_trajectory_absolute(video_path, output_jpg, output_pdf, N=10, blend_mode='colored_contours', abs_start=0.0, abs_end=None, mask_color=(255, 150, 0)):
    # Use absolute timestamps for trimming
    return visualize_trajectory_absolute_core(video_path, output_jpg, output_pdf, N, blend_mode, abs_start, abs_end, mask_color, is_absolute=True)

def visualize_trajectory_absolute_core(video_path, output_jpg, output_pdf, N=10, blend_mode='colored_contours', trim_start=0.0, trim_end=0.0, mask_color=(255, 150, 0), is_absolute=False):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found {video_path}")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return None, None, None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps if fps > 0 else 0
        
        start_frame = int(trim_start * fps)
        if is_absolute:
            # trim_end is actually the absolute end time
            end_time = trim_end if trim_end and trim_end > 0 else total_duration
            end_frame = min(total_frames, int(end_time * fps))
        else:
            # Traditional logic: trim_end means how many seconds to skip at the end
            end_frame = max(0, total_frames - int(trim_end * fps))
        
        if start_frame >= end_frame:
            print(f"❌ Error: Trim time (start:{trim_start}s, end:{trim_end}s) too long, resulting in insufficient valid video frames.")
            return None, None, None

        valid_frames = end_frame - start_frame

        if valid_frames < N:
            print(f"⚠️ Warning: Available video frames ({valid_frames}) is less than requested frames N ({N}). Adjusting N to {valid_frames}。")
            N = valid_frames

        print(f"Uniformly sampling {valid_frames} frames from a total of {N} frames (after trimming)...")
        
        frame_indices = np.linspace(start_frame, end_frame - 1, N, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32))
    finally:
        cap.release()

    if not frames:
        print("❌ Error: Failed to read any video frames.")
        return None, None, None

    print("Blending frames, this might take a moment...")

    layers_dict = None

    if blend_mode == 'average':
        blended = sum(frames) / len(frames)
    elif blend_mode == 'smart':
        background_u8 = np.median(frames, axis=0).astype(np.uint8)
        blended = background_u8.copy().astype(np.float32)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        for frame in frames:
            frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
            diff = cv2.absdiff(frame_u8, background_u8)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            alpha = mask.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            blended = blended * (1.0 - alpha) + frame * alpha
    elif blend_mode in ['colored_contours', 'focus_endpoints']:
        background_u8 = np.median(frames, axis=0).astype(np.uint8)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        
        base_color_uint8 = np.uint8([[[mask_color[0], mask_color[1], mask_color[2]]]])
        # To be safe, calculate RGB here for future use, bypassing OpenCV's built-in HSV logic
        RGB_base_color = mask_color 
        # print(f"[Core] 接收到的基础绘图目标 RGB_base_color 为: {RGB_base_color}")
        
        # Convert our received RGB to HSV. Must specify cv2.COLOR_RGB2HSV to get accurate Hue.
        base_hsv = cv2.cvtColor(base_color_uint8, cv2.COLOR_RGB2HSV)[0, 0]
        base_hue = int(base_hsv[0])
        # print(f"[Core] 提取到的基准 Hue 色相为: {base_hue} (范围 0-179)")
        
        num_frames = len(frames)
        history_alpha = []
        history_tint = []
        
        for i in range(num_frames):
            frame_u8 = np.clip(frames[i], 0, 255).astype(np.uint8)
            diff = cv2.absdiff(frame_u8, background_u8)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            
            _, mask_raw = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            contours_raw, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            min_area_threshold = 100
            if contours_raw:
                max_area = max(cv2.contourArea(c) for c in contours_raw)
                min_area_threshold = max(100, max_area * 0.05)
                
            for c in contours_raw:
                if cv2.contourArea(c) > min_area_threshold:
                    valid_contours.append(c)
            
            if not valid_contours:
                history_alpha.append(np.zeros((*frame_u8.shape[:2], 1), dtype=np.float32))
                history_tint.append(frame_u8.astype(np.float32))
                continue

            mask = np.zeros_like(mask_clean)
            cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill)
            mask_smooth = cv2.GaussianBlur(mask, (25, 25), 0)
            _, mask = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
            
            progress = i / max(1, num_frames - 1)
            
            if blend_mode == 'colored_contours':
                # No longer rely on messed up HSV calculations
                # Directly multiply R, G, B channels by percentage to create an effect from 'dim glow to bright main color'
                r = int(RGB_base_color[0] * (0.3 + 0.7 * progress))
                g = int(RGB_base_color[1] * (0.3 + 0.7 * progress))
                b = int(RGB_base_color[2] * (0.3 + 0.7 * progress))
                
                base_time_alpha = 0.2 + 0.8 * progress
                tint_intensity = 0.4
            else: 
                is_endpoint = (i == 0 or i == num_frames - 1)
                if is_endpoint:
                    r = RGB_base_color[0]
                    g = RGB_base_color[1]
                    b = RGB_base_color[2]
                    base_time_alpha = 1.0  
                    tint_intensity = 0.05  
                else:
                    r = int(RGB_base_color[0] * 0.5)
                    g = int(RGB_base_color[1] * 0.5)
                    b = int(RGB_base_color[2] * 0.5)
                    base_time_alpha = 0.5 
                    tint_intensity = 0.6   
            
            # Lock directly, no cvtColor function
            color_rgb = (r, g, b)
            # if i % (max(1, num_frames // 3)) == 0:
                # print(f"[Core] 渲染帧 {i+1}/{num_frames} -> 实际使用着色 RGB: {color_rgb}, tint: {tint_intensity:.2f}")

            mask_feathered = cv2.GaussianBlur(mask, (45, 45), 0)
            spatial_alpha = mask_feathered.astype(np.float32) / 255.0
            
            # Use correct numpy broadcasting assignment to fill the image. Previously using full_like(frame, tuple) forced all channels to the same value!
            frame_float = frame_u8.astype(np.float32)
            color_layer = np.empty_like(frame_float)
            color_layer[:] = color_rgb
            
            tint_weight = np.expand_dims(spatial_alpha * tint_intensity, axis=-1)
            tinted_float = frame_float * (1.0 - tint_weight) + color_layer * tint_weight
            
            final_alpha = np.expand_dims(spatial_alpha, axis=-1) * base_time_alpha
            
            history_alpha.append(final_alpha)
            history_tint.append(tinted_float)
            
        if blend_mode == 'focus_endpoints':
            mid_blended = background_u8.copy().astype(np.float32)
            for i in range(1, num_frames - 1):
                mid_blended = mid_blended * (1.0 - history_alpha[i]) + history_tint[i] * history_alpha[i]
                
            blended = mid_blended.copy()
            first_layer = frames[0].astype(np.uint8)  # Fully retain original first frame
            last_layer = frames[-1].astype(np.uint8)  # Fully retain original last frame
            
            if num_frames > 0:
                blended = blended * (1.0 - history_alpha[0]) + history_tint[0] * history_alpha[0]
                
            if num_frames > 1:
                blended = blended * (1.0 - history_alpha[-1]) + history_tint[-1] * history_alpha[-1]
                
            layers_dict = {
                'First Frame (Original)': first_layer,
                'Last Frame (Original)': last_layer,
                'Middle Trajectory Frame': np.clip(mid_blended, 0, 255).astype(np.uint8),
                'Clean Pure Background': background_u8.copy()
            }
        else:
            blended = background_u8.copy().astype(np.float32)
            for i in range(num_frames):
                blended = blended * (1.0 - history_alpha[i]) + history_tint[i] * history_alpha[i]
                
            layers_dict = {
                'First Frame (Original)': frames[0].astype(np.uint8),
                'Last Frame (Original)': frames[-1].astype(np.uint8),
                'Composite Image (All Trajectory)': np.clip(blended, 0, 255).astype(np.uint8),
                'Clean Pure Background': background_u8.copy()
            }
    elif blend_mode == 'max':
        blended = np.max(frames, axis=0)
    elif blend_mode == 'min':
        blended = np.min(frames, axis=0)
    else:
        blended = np.median(frames, axis=0)

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    img = Image.fromarray(blended)
    
    quality = 95
    while True:
        img.save(output_jpg, format='JPEG', quality=quality)
        file_size = os.path.getsize(output_jpg)
        if file_size <= 5 * 1024 * 1024 or quality <= 10:
            break
        quality -= 5
        
    img.save(output_pdf, format='PDF', resolution=100.0)
    return output_jpg, output_pdf, layers_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Trajectory Visualization Tool")
    parser.add_argument("video", type=str)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--blend", type=str, default="focus_endpoints")
    parser.add_argument("--color", type=str, default="255,150,0")
    parser.add_argument("--trim_start", type=float, default=0.0)
    parser.add_argument("--trim_end", type=float, default=0.0)
    
    args = parser.parse_args()
    if args.out is None:
        video_filename = os.path.basename(args.video)
        args.out = os.path.splitext(video_filename)[0] + "_trajectory"
        
    try:
        color_rgb = tuple(map(int, args.color.replace(' ', '').split(',')))
    except:
        color_rgb = (255, 150, 0)
    
    visualize_trajectory(args.video, f"{args.out}.jpg", f"{args.out}.pdf", args.n, args.blend, args.trim_start, args.trim_end, color_rgb)
