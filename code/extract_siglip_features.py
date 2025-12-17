import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

ROOT_DIR = r"C:\Users\wesle\Videos\PHOENIX-2014-T\features\fullFrame-210x260px"
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")

SIGLIP_ROOT = os.path.join(ROOT_DIR, "siglip_vitb16")
os.makedirs(SIGLIP_ROOT, exist_ok=True)

# --------------------------------------
# 1. Load SigLIP model (Google)
# --------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/siglip-base-patch16-224"
processor = SiglipImageProcessor.from_pretrained(model_name)
model = SiglipVisionModel.from_pretrained(model_name).to(device)
model.eval()

# --------------------------------------
# 2. Settings
# --------------------------------------
NUM_FRAMES = 180         # frames to sample per video
EMBED_DIM = 768         # SigLIP ViT-B/16 output

# --------------------------------------
# 3. Frame sampling
# --------------------------------------
def sample_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


# --------------------------------------
# 4. Extract SigLIP features
# --------------------------------------
def extract_siglip(video_path, out_path):
    if os.path.exists(out_path):
        return "skip"

    frames = sample_frames(video_path, NUM_FRAMES)
    if len(frames) == 0:
        np.save(out_path, np.zeros((0, EMBED_DIM), dtype=np.float32))
        return "empty"

    # Preprocess frames with SigLIP processor
    inputs = processor(images=frames, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pooled = outputs.pooler_output           # (F, 768)

    pooled = pooled.cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, pooled)
    return "ok"


# --------------------------------------
# 5. Process split
# --------------------------------------
def process_split(split):
    in_dir = os.path.join(VIDEOS_DIR, split)
    out_dir = os.path.join(SIGLIP_ROOT, split)

    if not os.path.isdir(in_dir):
        print(f"[WARN] {in_dir} not found")
        return

    os.makedirs(out_dir, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    print(f"[INFO] {split}: {len(videos)} videos")

    for v in tqdm(videos, desc=split):
        name = os.path.splitext(os.path.basename(v))[0]
        out_path = os.path.join(out_dir, f"{name}.npy")
        extract_siglip(v, out_path)


# --------------------------------------
# 6. Run extraction
# --------------------------------------
if __name__ == "__main__":
    # First test on dev
    #process_split("dev")

    # When confirmed:
    for split in ["train", "dev", "test"]:
        process_split(split)
