import time
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

# Load ViLT processor and model for Visual Question Answering (VQA)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, question):
    """
    Uses ViLT (Vision-and-Language Transformer) to answer a question about an image.
    
    Args:
        image (PIL.Image): Input image.
        question (str): Question related to the image.
    
    Returns:
        str: Predicted answer.
    """
    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

def type_out(text, delay=0.1):
    """
    Simulates a smooth typing effect for better user interaction.
    
    Args:
        text (str): Text to display.
        delay (float): Time delay between characters.
    """
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def main():
    """
    Loads an image, asks predefined questions, and allows users to input their own.
    """
    image_path = "/images/cat_dogs.jpg"  # Update with the correct image path
    image = Image.open(image_path)

    # Predefined questions
    questions = [
        "Where was this image taken?",
        "What do you see on the ground?",
        "Name the animals in this image.",
        "How many cats do you see in this image?",
        "What color is the cat?",
        "What color is the right dog?",
        "Which animal is the funniest in this image?"
    ]

    # Answer predefined questions with typing effect
    for question in questions:
        input("\nPress Enter to ask the next question...")
        type_out(question, delay=0.08)
        answer = answer_question(image, question)
        print(" →", answer)

    print("\n**** Now you can ask your own questions! Type 'exit' to quit. ****\n")

    # Interactive user-input loop
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting program. Have a great day!")
            break
        
        answer = answer_question(image, question)
        print(" →", answer)

if __name__ == "__main__":
    main()
