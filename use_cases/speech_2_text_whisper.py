import whisper
import sounddevice as sd
import numpy as np
import wave

def record_audio(filename="recorded_audio.wav", duration=5, samplerate=44100):
    """
    Records audio from the microphone and saves it as a WAV file.
    
    Args:
        filename (str): Output WAV file name.
        duration (int): Recording duration in seconds.
        samplerate (int): Sampling rate of the audio.
    """
    try:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        print("Recording finished.")

        # Save the recorded audio to a WAV file
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())

        print(f"Audio saved as {filename}")

    except Exception as e:
        print(f"Error recording audio: {e}")

def transcribe_audio(filename="recorded_audio.wav", model_size="small"):
    """
    Transcribes speech from an audio file using OpenAI's Whisper model.
    
    Args:
        filename (str): Path to the audio file.
        model_size (str): Whisper model size (e.g., 'tiny', 'base', 'small', 'medium', 'large').
    
    Returns:
        str: Transcribed text.
    """
    try:
        print(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)
        
        print(f"Transcribing {filename}...")
        result = model.transcribe(filename)
        
        print("Transcription complete.")
        return result["text"]

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def main():
    """
    Records audio and transcribes it using Whisper.
    """
    duration = 5  # Set recording duration (seconds)
    model_size = "small"  # Set Whisper model size
    
    record_audio(duration=duration)
    transcript = transcribe_audio(model_size=model_size)
    
    print("\n🔹 Transcribed Text:\n", transcript)

if __name__ == "__main__":
    main()
