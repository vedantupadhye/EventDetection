!pip install ultralytics
!pip install -U ultralytics
!pip install -U ultralytics

from ultralytics import YOLO
import glob
from google.colab import files
import os
# Load model
model = YOLO("yolo11m.pt")  # or use 'yolov8n.pt', 'yolov8s.pt', etc.

# Run ByteTrack on video
results = model.track(
    source="/content/walking.mp4",  # Make sure this video exists in /content/
    persist=True,
    stream=False,
    save=True,
    show=False,
    classes=None,  # Use [0] for person class only
    tracker="bytetrack.yaml",
    project='runs/track',  # Folder to save tracking results
    name='walking'         # Subfolder and video output name
)

# Search for the generated video
output_video = None
for root, dirs, files_list in os.walk("runs/track"):
    for file in files_list:
        if file.endswith(".mp4"):
            output_video = os.path.join(root, file)
            print("Found video:", output_video)

# Download the video
if output_video:
    files.download(output_video)
else:
    print("❌ No video found to download.")
