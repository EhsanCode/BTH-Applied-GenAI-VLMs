# BTH Applied Generative AI – Vision-Language Models (VLMs)  

This repository contains example codes and demos presented during the **Applied Generative AI** course at **BTH**. It covers **Vision-Language Models (VLMs)** and their practical applications in **computer vision, robotics, and speech processing**.  

## **Contents**  

### **Use Cases** (`use_cases/`)  
Example scripts for various VLM-based applications:  
- **Object Detection:** YOLOv8 for image and real-time camera detection.  
- **Vision-Language Models (VLMs):** BLIP, ViLT, and DETR using `transformers`.  
- **Speech Processing:** Speech-to-text (offline & Whisper), text-to-speech synthesis.  

### **Robotics** (`robotics/`)  
- **Object Recognition with a Robotic Arm:** Using VLMs to differentiate between an orange and an apple.
- The **robotic arm control function has been replaced with a simplified dummy function**, making the code easier to run for users who **do not have access to a real robot arm**.  
- The script supports **voice commands** to track and follow objects (apple/orange) using a camera.  
- It provides **two modes**:  
  1. **GPT-based mode** → Uses AI to interpret voice commands.  
  2. **Keyword-based mode** → Directly recognizes predefined keywords.

## **Setup & Installation**  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/yourusername/BTH-Applied-GenAI-VLMs.git
   cd BTH-Applied-GenAI-VLMs
   ```

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up the OpenAI API Key**  (Only for ```bash robotics/vlm_orange_or_apple.py```)

  Since the script requires OpenAI's API, set your key securely:  
  
  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  ```

## **How to Run**  

Run specific scripts based on the use case:  

- **Detect objects in an image using YOLOv8:**  
  ```bash
  python use_cases/detection_yolo8.py
  ```  

- **Detect objects in real-time using YOLOv8 with a camera:**  
  ```bash
  python use_cases/detection_yolo8_camera.py
  ```  

- **Run BLIP for Vision-Language tasks:**  
  ```bash
  python use_cases/pkg_transformers_Blip.py
  ```  

- **Perform speech-to-text conversion offline:**  
  ```bash
  python use_cases/speech_2_text_offline.py
  ```  

- **Convert text to speech:**  
  ```bash
  python use_cases/text_2_speech.py
  ```

- **How to Run the Object Tracking Demo**  

  Run the script and choose a mode:  
  ```bash
  python robotics/vlm_orange_or_apple.py
  ```

  - The program will ask you to **select between GPT-based and Keyword-based modes**.  
  - Speak into your microphone to **give a command** (e.g., *"Follow the orange"* or *"Stop following"*).  
  - The camera will detect and track objects **if a supported object (apple/orange) is detected**.  

  ### **Example Use Cases**  
  
  | **Task** | **Command Example** | **Expected Behavior** |
  |----------|--------------------|----------------------|
  | Track an apple | "Follow the apple" | The camera tracks the apple. |
  | Track an orange | "Can you follow the orange?" | The camera tracks the orange. |
  | Stop tracking | "Stop" or "End tracking" | Tracking stops. |


## **Example Images**  
Sample images used in the experiments can be found in the `use_cases/images/` folder.  

## **References & Resources**  
- [YOLOv8 Documentation](https://github.com/ultralytics/ultralytics)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
- [OpenAI Whisper](https://openai.com/research/whisper)  

