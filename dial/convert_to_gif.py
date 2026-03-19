#!/usr/bin/env python3
"""Convert real-world demo videos to 8:10 GIFs at 4x speed."""
import cv2, os, glob
from PIL import Image

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos", "real_world_processed")
DST = "/Users/cheny/Documents/PhD/papers/NIPS_2026/figs/real_world_processed_gif"
W_OUT = 240
H_OUT = 192
FRAME_SKIP = 8
FRAME_DURATION_MS = 67

os.makedirs(DST, exist_ok=True)
videos = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
print(f"Converting {len(videos)} videos to {W_OUT}x{H_OUT} GIFs at 4x speed...\n")

for vpath in videos:
    name = os.path.splitext(os.path.basename(vpath))[0]
    dst_path = os.path.join(DST, name + ".gif")

    cap = cv2.VideoCapture(vpath)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sq = cv2.resize(frame_rgb, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(sq)
            frames.append(img)
        idx += 1
    cap.release()

    if not frames:
        print(f"  {name}: no frames, skipping")
        continue

    frames[0].save(
        dst_path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )

    sz = os.path.getsize(dst_path) / 1024 / 1024
    print(f"  {name}: {len(frames)} frames -> {sz:.1f}MB", flush=True)

print("\nDone!")
