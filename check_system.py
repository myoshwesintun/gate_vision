#!/usr/bin/env python3
"""Basic pre-flight checks for Gate Vision on Raspberry Pi."""
from pathlib import Path
import importlib.util
import shutil
import sys

BASE = Path(__file__).resolve().parent
FILES = BASE / "files"

checks = []

def add(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

for module in ["cv2", "customtkinter", "PIL", "pytesseract", "ultralytics", "numpy"]:
    add(f"Python module: {module}", importlib.util.find_spec(module) is not None)

add("Tesseract executable", shutil.which("tesseract") is not None, shutil.which("tesseract") or "not found")

for filename in ["license_plate_yolo11n.pt", "registered_vehicles.csv", "TUTGO.png", "EC.png"]:
    path = FILES / filename
    add(f"File: {filename}", path.is_file(), str(path))

try:
    import cv2
    camera_ok = False
    camera_idx = None
    for idx in (0, 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                camera_ok = True
                camera_idx = idx
                cap.release()
                break
        cap.release()
    add("Camera", camera_ok, f"index {camera_idx}" if camera_ok else "no frame from index 0/1")
except Exception as exc:
    add("Camera", False, str(exc))

try:
    import gpiozero
    add("gpiozero", True)
except Exception as exc:
    add("gpiozero", False, str(exc))

print("\nGate Vision system check")
print("=" * 60)
for name, ok, detail in checks:
    status = "PASS" if ok else "WARN"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status:4}] {name}{suffix}")

critical = [ok for name, ok, _ in checks if name.startswith("File:") or name.startswith("Python module:")]
if not all(critical):
    sys.exit(1)
