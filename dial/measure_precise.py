#!/usr/bin/env python3
"""Generate zoomed inspection crops of head and table regions for precise landmark measurement."""

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

for key in VIDEOS:
    path = os.path.join(SRC_DIR, key + ".mp4")
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        continue
    h, w = frame.shape[:2]

    # Head region: 5% to 30% of frame height, with 1% grid
    head_y1 = int(0.05 * h)
    head_y2 = int(0.30 * h)
    head_crop = frame[head_y1:head_y2, :].copy()
    ch = head_crop.shape[0]
    for pct_10 in range(5, 31):
        y_abs = int(pct_10 / 100.0 * h)
        y_rel = y_abs - head_y1
        if 0 <= y_rel < ch:
            color = (0, 0, 255) if pct_10 % 5 == 0 else (0, 200, 255)
            thickness = 2 if pct_10 % 5 == 0 else 1
            cv2.line(head_crop, (0, y_rel), (w, y_rel), color, thickness)
            cv2.putText(head_crop, f"{pct_10}%", (5, y_rel - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # Table region: 48% to 75% of frame height, with 1% grid
    table_y1 = int(0.48 * h)
    table_y2 = int(0.75 * h)
    table_crop = frame[table_y1:table_y2, :].copy()
    ct = table_crop.shape[0]
    for pct_10 in range(48, 76):
        y_abs = int(pct_10 / 100.0 * h)
        y_rel = y_abs - table_y1
        if 0 <= y_rel < ct:
            color = (0, 0, 255) if pct_10 % 5 == 0 else (0, 200, 255)
            thickness = 2 if pct_10 % 5 == 0 else 1
            cv2.line(table_crop, (0, y_rel), (w, y_rel), color, thickness)
            cv2.putText(table_crop, f"{pct_10}%", (5, y_rel - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    combined = np.vstack([head_crop, np.ones((4, w, 3), dtype=np.uint8) * 255, table_crop])
    cv2.putText(combined, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    out_path = os.path.join(INSPECT_DIR, f"{key}_zones.jpg")
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{key}: head zone [{head_y1}-{head_y2}], table zone [{table_y1}-{table_y2}]")

# Also do a programmatic scan for approximate landmarks
print("\n--- Programmatic landmark estimates ---")
for key in VIDEOS:
    path = os.path.join(SRC_DIR, key + ".mp4")
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        continue
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Head top: scan from top, find first row with significant dark pixels (robot head/cap)
    center_strip = gray[:, w//3:2*w//3]  # center third
    head_top_y = None
    for y in range(int(0.05 * h), int(0.30 * h)):
        row = center_strip[y]
        dark_count = np.sum(row < 60)
        if dark_count > 30:
            head_top_y = y
            break

    # Table top: scan downward from 45%, find where mean brightness in center strip jumps above 200
    table_top_y = None
    for y in range(int(0.45 * h), int(0.75 * h)):
        row_mean = np.mean(center_strip[y])
        if row_mean > 190:
            table_top_y = y
            break

    # Table limit: scan from table_top downward, find where mean drops significantly (under-table region)
    table_limit_y = None
    if table_top_y:
        for y in range(table_top_y + 10, min(int(0.80 * h), h)):
            row_mean = np.mean(center_strip[y])
            if row_mean < 150:
                table_limit_y = y
                break

    ht = head_top_y / h if head_top_y else None
    tt = table_top_y / h if table_top_y else None
    tl = table_limit_y / h if table_limit_y else None
    print(f"{key:45s}  head_top={ht:.3f}  table_top={tt:.3f}  table_limit={tl:.3f}" if all([ht, tt, tl]) else f"{key}: detection failed")
