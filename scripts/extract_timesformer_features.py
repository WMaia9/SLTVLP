import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from transformers import AutoImageProcessor, TimesformerModel

# Root and dirs
ROOT_DIR = r"C:\Users\wesle\Music\PHOENIX14T"
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")

# Renaming folder to reflect sliding window approach
TIMEFORMER_ROOT = os.path.join(ROOT_DIR, "timesformer_sliding_window") 
os.makedirs(TIMEFORMER_ROOT, exist_ok=True)

# --------------------------------------
# 1. Load TimeSformer model
# --------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/timesformer-base-finetuned-k400"

print("Loading model...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = TimesformerModel.from_pretrained(model_name).to(device)
model.eval()

# TimeSformer expects exactly 8 frames by default for this specific HF model checkpoint
# If you force 32, it might interpolate, but 8 is the native training resolution.
WINDOW_SIZE = 8   
STRIDE = 4        # 50% overlap to capture smooth motion
HIDDEN_SIZE = model.config.hidden_size 

print(f"Using device: {device}")
print(f"Window Size: {WINDOW_SIZE}, Stride: {STRIDE}")

# --------------------------------------
# 2. Load ALL Frames (No sparse sampling)
# --------------------------------------
def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# --------------------------------------
# 3. Sliding Window Extraction
# --------------------------------------
def extract_timesformer_features(video_path, out_path):
    if os.path.exists(out_path):
        return "skip"

    all_frames = load_all_frames(video_path)
    T = len(all_frames)
    
    if T < WINDOW_SIZE:
        # Handle very short videos: Pad or duplicate
        if T == 0: return "empty"
        all_frames = all_frames + [all_frames[-1]] * (WINDOW_SIZE - T)
        T = len(all_frames)

    # Create windows
    # E.g. [0-8], [4-12], [8-16]...
    window_features = []
    
    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        clip = all_frames[start:end]
        
        # Processor expects list of images
        inputs = processor(images=clip, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device) # (1, 8, 3, 224, 224)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # Extract [CLS] token (Index 0) which summarizes the motion in this window
            # shape: (1, 197, 768) -> take index 0 -> (1, 768)
            cls_token = outputs.last_hidden_state[:, 0, :]
            
        window_features.append(cls_token.cpu().numpy())

    # Check if we have features
    if len(window_features) == 0:
        # Fallback for edge cases
        return "empty"

    # Stack to get (Num_Windows, 768)
    video_feat_seq = np.concatenate(window_features, axis=0) # (T_new, 768)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, video_feat_seq)
    return "ok"

# --------------------------------------
# 4. Process one split
# --------------------------------------
def process_split(split):
    in_dir = os.path.join(VIDEOS_DIR, split)
    out_dir = os.path.join(TIMEFORMER_ROOT, split)

    if not os.path.isdir(in_dir):
        print(f"[WARN] {in_dir} not found")
        return

    os.makedirs(out_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    print(f"[INFO] {split}: {len(videos)} videos")

    for v in tqdm(videos, desc=split):
        name = os.path.splitext(os.path.basename(v))[0]
        out_path = os.path.join(out_dir, f"{name}.npy")
        extract_timesformer_features(v, out_path)

if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        process_split(split)