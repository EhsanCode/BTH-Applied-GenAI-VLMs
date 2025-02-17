import time
import speech_recognition as sr
import cv2
from PIL import Image
from ultralytics import YOLO
import pyttsx3
from openai import OpenAI


# Load YOLOv8 model (pre-trained on COCO dataset)
model_yolo = YOLO("yolov8l.pt")

# Initialize speech recognizer
speech_recognizer = sr.Recognizer()

# Define trackable objects
trackable_objects = ["apple", "orange"]


def move_robot_arm(x, y):
    """ Dummy function to simulate robot arm movement. """
    print(f"Moving robot arm to coordinates: ({x}, {y})")


def say(text):
    """ Convert text to speech using pyttsx3. """
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()
    print("Robot says:", text)


def get_command(message):
    """ Generate a response using GPT-based AI model. """
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message
    )
    return completion.choices[0].message.content


def get_object_detections(frame, object_name):
    """ Detect specified objects in a video frame using YOLOv8. """
    detections = []
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Run YOLOv8 inference
    results = model_yolo(image)
    for result in results:
        for box, label, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            rec_color = (125, 0, 0)
            detected_object_name = model_yolo.names[int(label)]
            x1, y1, x2, y2 = map(int, box.tolist())

            if detected_object_name == object_name:
                detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'score': float(score)})
                rec_color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), rec_color, 2)
            cv2.putText(frame, f"{detected_object_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return sorted(detections, key=lambda x: -x['score'])


def track_apple_or_orange(object_name, following_time=30):
    """ Tracks an apple or orange using the camera for a set duration. """
    image_center = [640 / 2, 480 / 2]
    cap = cv2.VideoCapture(0)  # Change camera index if needed

    start_time = time.time()
    while (time.time() - start_time) < following_time:
        ret, frame = cap.read()
        if not ret:
            break

        detections = get_object_detections(frame, object_name)
        if detections:
            center = [detections[0]['x1'] + (detections[0]['x2'] - detections[0]['x1']) // 2 - image_center[0],
                      detections[0]['y1'] + (detections[0]['y2'] - detections[0]['y1']) // 2 - image_center[1]]
            move_robot_arm(center[0], center[1])

        cv2.imshow("Detections", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def listener():
    """ Listens to a voice command and converts it to text using Whisper. """
    with sr.Microphone() as source:
        print("Listening for a command...")
        audio = speech_recognizer.listen(source)

    try:
        text = speech_recognizer.recognize_whisper(audio, language="english")
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Error with Whisper API.")
    return ""


def get_instruction():
    """ Provides a structured instruction set for GPT-based command parsing. """
    return [{
        "role": "user",
        "content": ("A robot listens to a voice command and can either follow an object (apple/orange) or stop. "
                    "Convert any given command into one of the following words: "
                    "'apple', 'orange', or 'stop'. If the command is unclear, return 'error'.")
    }, {
        "role": "system",
        "content": "OK"
    }]


def gpt_based_demo():
    """ GPT-based demo for object tracking via voice commands. """
    instruction = get_instruction()

    while True:
        prompt = listener()
        if not prompt:
            continue

        response = get_command(instruction + [{"role": "user", "content": prompt}])

        if response == "stop":
            break
        if response in trackable_objects:
            say(f"I'm going to follow, {response}.")
            track_apple_or_orange(response)


def keyword_based_demo():
    """ Keyword-based demo for object tracking via simple voice commands. """
    while True:
        command = listener()
        if not command:
            continue

        if "stop" in command:
            break

        for object_name in trackable_objects:
            if object_name in command:
                track_apple_or_orange(object_name)
                return

        print(f"Please say either 'apple' or 'orange' to track.")


def main():
    while True:
        print("Select mode:")
        print("1 - Keyword-based demo")
        print("2 - GPT-based demo")
        print("3 - Ext")
        choice = input("Enter choice (1/2): ").strip()

        if choice == "1":
            keyword_based_demo()
        elif choice == "2":
            gpt_based_demo()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
