# Cow Detection in Videos

This repository contains a simple script to detect cows in a video and draw bounding boxes around them using a YOLOv8 model.

## Requirements

- Python 3.8+
- [ultralytics](https://docs.ultralytics.com/) library (provides YOLOv8)
- `opencv-python`

Install the dependencies with:

```bash
pip install ultralytics opencv-python
```

## Usage

Run the script providing a path to the input video and an optional output path. The script will display the detection in real time and save the processed video.

```bash
python detect_cows_video.py --source path/to/video.mp4 --output output.mp4
```

Press `q` in the display window to stop early.

The script uses the YOLOv8 model pre-trained on the COCO dataset, which includes a `cow` class (index `20`).
