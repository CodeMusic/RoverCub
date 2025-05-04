import RPi.GPIO as GPIO
import time
import speech_recognition as sr
from gtts import gTTS
import os

# Setup GPIO
BUTTON = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Setup recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=None)  # Use default or specify your device index

print("Assistant ready. Hold the button to speak.")

def record_audio():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        print("Processing...")
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return None
    except sr.RequestError as e:
        print(f"Request error: {e}")
        return None

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("/tmp/response.mp3")
    os.system("mpg123 /tmp/response.mp3")

try:
    while True:
        if GPIO.input(BUTTON) == 0:  # Button pressed
            time.sleep(0.1)  # Debounce
            if GPIO.input(BUTTON) == 0:
                captured_text = record_audio()
                if captured_text:
                    # Example: respond simply
                    response_text = f"You said: {captured_text}"
                    speak(response_text)
            time.sleep(0.5)  # Avoid double captures

except KeyboardInterrupt:
    print("Exiting...")
    GPIO.cleanup()
