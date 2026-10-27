"""Generates a small synthetic case (image, transcript, video) for demos/tests."""
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEMO_TRANSCRIPT = """[00:02] This is Officer Rana. It is 11:42 PM near the Central Market parking lot.
[00:05] Witness: I saw two men arguing behind the shop.
[00:08] One of them, a tall man in a red jacket, pulled out a knife.
[00:10] He pushed the other man and then ran toward Station Road.
[00:13] The victim was left on the ground, shouting for help.
"""


def make_image(path, size=(640, 400)):
    img = Image.new("RGB", size, (28, 32, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 280, size[0], size[1]], fill=(46, 50, 58))
    for x, col in ((200, (150, 40, 40)), (330, (60, 80, 120))):
        d.ellipse([x, 130, x + 40, 170], fill=(200, 170, 140))
        d.rectangle([x - 10, 170, x + 50, 290], fill=col)
    d.rectangle([520, 60, 600, 120], fill=(230, 220, 120))
    img.save(path)


def make_video(path, seconds=14, fps=10, size=(480, 360)):
    """Two moving figures; violent jitter + flash between 8s and 11s."""
    rnd = random.Random(7)
    codecs = [("VP80", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4")]
    writer, final = None, None
    for fourcc, ext in codecs:
        target = path.with_suffix(ext)
        w = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            writer, final = w, target
            break
    if writer is None:
        raise RuntimeError("No usable video codec in this OpenCV build")
    pos = [[120.0, 120.0], [300.0, 130.0]]
    for i in range(seconds * fps):
        t = i / fps
        frame = np.full((size[1], size[0], 3), (40, 44, 52), np.uint8)
        cv2.rectangle(frame, (0, 270), (size[0], size[1]), (58, 62, 70), -1)
        violent = 8 <= t < 11
        for n, p in enumerate(pos):
            step = rnd.uniform(-45, 45) if violent else rnd.uniform(-1.2, 1.2)
            p[0] = float(np.clip(p[0] + step, 40, size[0] - 80))
            p[1] = float(np.clip(p[1] + (rnd.uniform(-25, 25) if violent else 0), 90, 150))
            x, y = int(p[0]), int(p[1])
            cv2.circle(frame, (x + 20, y), 18, (170, 190, 220), -1)
            cv2.rectangle(frame, (x, y + 18), (x + 40, y + 120), (40, 40, 160) if n == 0 else (120, 90, 50), -1)
        if violent and i % 5 == 0:
            frame = cv2.add(frame, np.full_like(frame, 60))
        cv2.putText(frame, f"CAM-02  00:{int(t):02d}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        writer.write(frame)
    writer.release()
    return final
