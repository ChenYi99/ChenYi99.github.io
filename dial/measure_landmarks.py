#!/usr/bin/env python3
"""Extract first frames from source videos and generate annotated images for landmark measurement."""

import cv2
import numpy as np
import os

SRC_DIR = "/Users/cheny/Documents/PhD/papers/NIPS_2026/project_page/videos_raw"
INSPECT_DIR = "/Users/cheny/Documents/PhD/papers/NIPS_2026/DIAL_project_page/videos/real_world_processed/_inspect"

VIDEOS = [
    "id_pour_pepsi", "id_pour_water", "id_put_banana", "id_put_bowl",
    "ood_combination_put_banana", "ood_combination_put_bowl",
    "ood_combination_put_bowl_single_side", "ood_distractor_banana",
    "ood_distractor_bowl", "ood_pour_mirinda", "ood_pour_tea", "ood_pour_water",
]

os.makedirs(INSPECT_DIR, exist_ok=True)

frames = {}
for key in VIDEOS:
    path = os.path.join(SRC_DIR, key + ".mp4")
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: cannot read {path}")
        continue
    frames[key] = frame
    h, w = frame.shape[:2]
    print(f"{key}: {w}x{h}")

    annotated = frame.copy()
    for pct in range(0, 100, 2):
        y = int(pct / 100.0 * h)
        color = (0, 0, 255) if pct % 10 == 0 else (180, 180, 180)
        thickness = 2 if pct % 10 == 0 else 1
        cv2.line(annotated, (0, y), (w, y), color, thickness)
        cv2.putText(annotated, f"{pct}%", (5, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    out_path = os.path.join(INSPECT_DIR, f"{key}_grid.jpg")
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Build a compact overview grid: 4 columns
cols = 4
thumb_w = 360
keys_sorted = sorted(frames.keys())
thumbs = []
for key in keys_sorted:
    f = frames[key]
    h, w = f.shape[:2]
    thumb_h = int(thumb_w * h / w)
    thumb = cv2.resize(f, (thumb_w, thumb_h))
    for pct in range(0, 100, 5):
        y = int(pct / 100.0 * thumb_h)
        color = (0, 0, 255) if pct % 10 == 0 else (200, 200, 200)
        thickness = 2 if pct % 10 == 0 else 1
        cv2.line(thumb, (0, y), (thumb_w, y), color, thickness)
        if pct % 10 == 0:
            cv2.putText(thumb, f"{pct}%", (3, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(thumb, key, (5, thumb_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    thumbs.append(thumb)

rows_needed = (len(thumbs) + cols - 1) // cols
while len(thumbs) < rows_needed * cols:
    thumbs.append(np.zeros_like(thumbs[0]))

grid_rows = []
for r in range(rows_needed):
    grid_rows.append(np.hstack(thumbs[r * cols:(r + 1) * cols]))
grid = np.vstack(grid_rows)

grid_path = os.path.join(INSPECT_DIR, "_source_overview.jpg")
cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nOverview grid saved to {grid_path} ({grid.shape[1]}x{grid.shape[0]})")
print("Individual annotated frames saved to _inspect/ directory.")
