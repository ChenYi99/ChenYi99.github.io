#!/usr/bin/env python3
"""Convert egodex and iron_ego videos to GIF with dataset-specific cropping."""
import cv2, os, glob
from PIL import Image

BASE = "/Users/cheny/Documents/PhD/papers/NIPS_2026"
FIGS = os.path.join(BASE, "figs")
FRAME_SKIP = 2
OUT_SIZE = 240


def make_gif(frames, dst_path, duration_ms):
    if not frames:
        return
    frames[0].save(
        dst_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    sz = os.path.getsize(dst_path) / 1024 / 1024
    print(f"  {os.path.basename(dst_path)}: {len(frames)} frames, {duration_ms}ms/frame -> {sz:.1f}MB", flush=True)


def process_egodex():
    src_dir = os.path.join(BASE, "videos", "egodex")
    dst_dir = os.path.join(FIGS, "egodex_gif")
    os.makedirs(dst_dir, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
    print(f"=== egodex: {len(videos)} videos -> {OUT_SIZE}x{OUT_SIZE} ===\n")

    for vpath in videos:
        name = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration_ms = int(FRAME_SKIP * 1000 / fps)

        crop_size = min(w, h)  # 1080
        if name == "episode_001004":
            x_off = w - crop_size  # 840, keep right side
        else:
            x_off = (w - crop_size) // 2  # 420, center crop

        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % FRAME_SKIP == 0:
                cropped = frame[0:crop_size, x_off:x_off + crop_size]
                resized = cv2.resize(cropped, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1
        cap.release()

        make_gif(frames, os.path.join(dst_dir, name + ".gif"), duration_ms)


def process_iron_ego():
    src_dir = os.path.join(BASE, "videos", "iron_ego")
    dst_dir = os.path.join(FIGS, "iron_ego_gif")
    os.makedirs(dst_dir, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
    print(f"\n=== iron_ego: {len(videos)} videos -> {OUT_SIZE}x{OUT_SIZE} ===\n")

    for vpath in videos:
        name = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration_ms = int(FRAME_SKIP * 1000 / fps)

        y1 = h // 2 + h // 20   # 594, remove top half + extra 1/20
        y2 = h                   # 1080
        x1 = w // 6              # 320, remove left 1/6
        x2 = w - w // 6          # 1600, remove right 1/6

        if name == "episode_000053":
            x1 += w // 10        # extra 1/10 from left
            y1 += h // 20 + h // 10  # extra 1/20 + 1/10 from top
            # y2 -= h // 20 + h // 10  # 1/20 + 1/10 less from bottom
        elif name == "episode_000026":
            x1 += w // 10        # extra 1/10 from left
        elif name in ("segment_0_front_right", "segment_1_front_right"):
            x2 -= w // 10        # extra 1/10 from right

        skip_frames = 0
        if name == "segment_0_front_right":
            skip_frames = int(fps * 4)  # skip first 4 seconds

        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < skip_frames:
                idx += 1
                continue
            if (idx - skip_frames) % FRAME_SKIP == 0:
                cropped = frame[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1
        cap.release()

        make_gif(frames, os.path.join(dst_dir, name + ".gif"), duration_ms)


if __name__ == "__main__":
    process_egodex()
    process_iron_ego()
    print("\nDone!")
