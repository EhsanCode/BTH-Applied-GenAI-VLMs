from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor for image captioning
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    """
    Generates a caption for the given image using BLIP.
    
    Args:
        image (PIL.Image): Input image.
    
    Returns:
        str: Generated caption text.
    """
    inputs = processor(images=image, return_tensors="pt")  # Preprocess image
    with torch.no_grad():  # Disable gradient computation for inference
        caption_ids = model.generate(**inputs)  # Generate caption

    return processor.batch_decode(caption_ids, skip_special_tokens=True)[0]  # Decode output

def main():
    """
    Loads an image and generates a caption.
    """
    image_path = "city.jpeg"  # Update with your image path
    image = Image.open(image_path)
    
    caption = generate_caption(image)
    print("Caption:", caption)

if __name__ == "__main__":
    main()
