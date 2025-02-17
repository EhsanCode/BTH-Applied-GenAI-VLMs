import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load DETR model and processor for object detection
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def detect_objects(image, threshold=0.9):
    """
    Detects objects in an image using DETR (DEtection TRansformer).
    
    Args:
        image (PIL.Image): Input image for detection.
        threshold (float): Confidence threshold for filtering detections.
    
    Returns:
        dict: Dictionary containing detected bounding boxes, labels, and scores.
    """
    inputs = processor(images=image, return_tensors="pt")  # Preprocess image
    with torch.no_grad():
        outputs = model(**inputs)  # Run inference

    # Convert target sizes from (width, height) to (height, width) format
    target_sizes = torch.tensor([image.size[::-1]])
    
    # Process output detections
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    return results

def plot_detections(image, results):
    """
    Plots the detected objects with bounding boxes on the image.
    
    Args:
        image (PIL.Image): Original image.
        results (dict): Object detection results containing bounding boxes, labels, and scores.
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    # Extract bounding boxes, labels, and scores
    bboxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    # Draw bounding boxes
    for bbox, label, score in zip(bboxes, labels, scores):
        label_name = model.config.id2label[label.item()]
        score_value = score.item()
        x, y, w, h = bbox.tolist()

        # Draw rectangle
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

        # Add label and confidence score
        label_text = f"{label_name} ({score_value:.2f})"
        ax.text(x, y - 5, label_text, fontsize=10, color="white",
                bbox=dict(facecolor="red", alpha=0.5))

        print(f"Detected: {label_name} | BBox: {bbox.tolist()} | Confidence: {score_value:.2f}")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def main():
    """
    Loads an image, runs DETR object detection, and visualizes the results.
    """
    image_path = "/images/cat_dogs.jpg"  # Change to your image file
    image = Image.open(image_path)

    results = detect_objects(image, threshold=0.9)  # Run object detection
    plot_detections(image, results)  # Display results

if __name__ == "__main__":
    main()
