import vosk
import sys
import json
import pyaudio

def initialize_recognizer(model_path="vosk-model-en-us-0.22"):
    """
    Initializes the Vosk speech recognizer and microphone.
    
    Args:
        model_path (str): Path to the Vosk model directory.
    
    Returns:
        tuple: Recognizer object and microphone stream.
    """
    print("Loading Vosk model... This may take a few seconds.")
    
    # Load speech recognition model
    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, 16000)

    # Initialize microphone
    try:
        mic = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        mic.start_stream()
        return recognizer, mic
    except Exception as e:
        print("Error initializing microphone:", e)
        sys.exit(1)

def listen_and_recognize(recognizer, mic):
    """
    Continuously listens for speech input and prints recognized text.
    
    Args:
        recognizer (KaldiRecognizer): Vosk recognizer instance.
        mic (PyAudio Stream): Audio stream for input.
    """
    print("Listening... (Say 'exit' to stop)")

    while True:
        data = mic.read(4000, exception_on_overflow=False)
        
        if recognizer.AcceptWaveform(data):
            result_text = json.loads(recognizer.Result())["text"]
            print("You said:", result_text)

            # Exit the program if the user says "exit"
            if "exit" in result_text.lower():
                print("Exiting speech recognition...")
                break

def main():
    """
    Main function to initialize the recognizer and start listening.
    """
    recognizer, mic = initialize_recognizer()
    listen_and_recognize(recognizer, mic)

if __name__ == "__main__":
    main()
