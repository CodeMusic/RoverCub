import RPi.GPIO as GPIO
import time
import speech_recognition as sr
import openai
import tempfile
import os
from gtts import gTTS

# GPIO pin setup
BUTTON = 17
LED_RED = 27
LED_GREEN = 22

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_RED, GPIO.OUT)
GPIO.setup(LED_GREEN, GPIO.OUT)

# Turn on green by default
GPIO.output(LED_GREEN, GPIO.HIGH)
GPIO.output(LED_RED, GPIO.LOW)

# Setup mic and recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=None)

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI

client = OpenAI()

def record_audio():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        print("Recording complete.")

    # Save audio to WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio.get_wav_data())
        temp_path = temp_file.name

    try:
        print("Sending to OpenAI Whisper...")
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        print("You said:", transcript)
        return transcript
    except Exception as e:
        print("Whisper error:", e)
        return None

def speak(text):
    print("Speaking:", text)
    tts = gTTS(text=text, lang='en')
    tts.save("/tmp/reply.mp3")
    os.system("mpg123 /tmp/reply.mp3")

try:
    print("Assistant ready. Hold the button and speak.")
    while True:
        if GPIO.input(BUTTON) == GPIO.LOW:
            print("Button pressed!")
            time.sleep(0.1)  # Debounce
            if GPIO.input(BUTTON) == GPIO.LOW:
                user_text = record_audio()
                if user_text:
                    reply = f"You said: {user_text}"
                    speak(reply)
            time.sleep(0.5)  # Delay before allowing next press

except KeyboardInterrupt:
    print("\nExiting...")
    GPIO.cleanup()
