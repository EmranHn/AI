import argparse
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Detect cows in video using YOLO")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Path to output video")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    cow_class = 20  # COCO class index for cow

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box, cls_id, conf in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf
        ):
            if int(cls_id) == cow_class:
                x1, y1, x2, y2 = map(int, box)
                label = f"cow {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        out.write(frame)
        cv2.imshow("Cow detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
