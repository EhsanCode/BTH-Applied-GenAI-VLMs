import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8l.pt")  # Available versions: 'n' (smallest), 's', 'm', 'l' (largest)

# Load and preprocess the image
image_path = "frame_.png"  # Change this to your image file path
image = cv2.imread(image_path)

# Ensure the image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Error: Unable to load image at '{image_path}'.")

# Convert image from BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run YOLOv8 inference on the image
results = model(image)

# Set up the figure for visualization
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.axis("off")  # Hide axis labels

# Process detection results
for result in results:
    boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
    labels = result.boxes.cls  # Class IDs
    scores = result.boxes.conf  # Confidence scores

    for box, label, score in zip(boxes, labels, scores):
        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, box.tolist())

        # Get class name and confidence score
        class_name = model.names[int(label)]
        label_text = f"{class_name} ({score:.2f})"

        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False
        ))

        # Add label above the bounding box
        plt.text(x1, y1 - 5, label_text, color='red', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5))

# Display the image with detections
plt.show()
