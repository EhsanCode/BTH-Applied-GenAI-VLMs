import cv2
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8l.pt")  # Available sizes: 'n', 's', 'm', 'l' (smallest to largest)

def camera_object_detection():
    """Real-time object detection using YOLOv8 and webcam."""
    cap = cv2.VideoCapture(0)  # Open webcam (change index if needed)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Convert frame from BGR to RGB (for PIL processing)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run YOLO inference
        results = model(image)

        # Process detections
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            labels = result.boxes.cls  # Class IDs
            scores = result.boxes.conf  # Confidence scores

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box.tolist())  # Convert to integers
                label_text = f"{model.names[int(label)]} ({score:.2f})"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("YOLO Object Detection", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup: Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_object_detection()
