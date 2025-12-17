import os
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from extract_keypoints_core import extract_keypoints_best  # adjust import if needed


ROOT_DIR = r"C:\Users\wesle\Videos\PHOENIX-2014-T\features\fullFrame-210x260px"
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
KPTS_DIR   = os.path.join(ROOT_DIR, "kpts")

os.makedirs(KPTS_DIR, exist_ok=True)


def process_one_video(video_path: str, split: str):
    """
    Extract keypoints for a single video and save as .npy.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]  # e.g., 01April_2010_Thursday_heute-6694
    out_split_dir = os.path.join(KPTS_DIR, split)
    os.makedirs(out_split_dir, exist_ok=True)
    out_path = os.path.join(out_split_dir, basename + ".npy")

    if os.path.exists(out_path):
        # Skip if already processed
        return basename, "skip"

    kpts = extract_keypoints_best(video_path)
    np.save(out_path, kpts)
    return basename, "ok"


def process_split(split: str, num_workers: int = 12):
    """
    Process one split (train/dev/test) in parallel.
    """
    split_video_dir = os.path.join(VIDEOS_DIR, split)
    if not os.path.isdir(split_video_dir):
        print(f"[WARN] No video directory for split '{split}': {split_video_dir}")
        return

    pattern = os.path.join(split_video_dir, "*.mp4")
    videos = sorted(glob.glob(pattern))
    total = len(videos)
    print(f"[INFO] Split '{split}': found {total} videos")

    if total == 0:
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one_video, v, split): v for v in videos}
        done = 0
        for fut in as_completed(futures):
            done += 1
            video_path = futures[fut]
            try:
                name, status = fut.result()
                print(f"[{split}] {done}/{total} -> {name} [{status}]")
            except Exception as e:
                print(f"[ERROR] {split}: error processing {video_path}: {e}")


if __name__ == "__main__":
    # You can tune this: 12–16 is reasonable for your 12C/24T machine with NVMe
    NUM_WORKERS = 12

    # Start with a small split to test:
    #for sp in ["dev"]:
    #    process_split(sp, num_workers=NUM_WORKERS)

    # When dev is OK, run:
    for sp in ["train", "dev", "test"]:
        process_split(sp, num_workers=NUM_WORKERS)