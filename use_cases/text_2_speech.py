import pyttsx3


def say(text):
    # Convert text to speech
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()
    print("Robo: ", text)


say("Hello world!")