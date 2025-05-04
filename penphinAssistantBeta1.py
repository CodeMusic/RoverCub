import RPi.GPIO as GPIO
import time
import speech_recognition as sr
import openai
import tempfile
import os
from gtts import gTTS
from typing import List, Dict, Optional
import requests
import json

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

class LLMConfig:
    def __init__(
        self,
        use_local: bool = True,
        local_endpoint: str = "http://172.20.10.3:1234/v1",
        cloud_api_key: Optional[str] = None,
        system_message: str = "You are a helpful AI assistant. Keep responses concise and conversational.",
        max_tokens: int = 150
    ):
        self.use_local = use_local
        self.local_endpoint = local_endpoint
        self.cloud_api_key = cloud_api_key
        self.system_message = system_message
        self.max_tokens = max_tokens
        
        if not use_local and cloud_api_key:
            openai.api_key = cloud_api_key
            self.client = openai.OpenAI()
        else:
            self.client = None

class ConversationManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        
    def add_user_message(self, message: str) -> None:
        self.conversation_history.append({"role": "user", "content": message})
        
    def add_assistant_message(self, message: str) -> None:
        self.conversation_history.append({"role": "assistant", "content": message})
        
    def get_llm_response(self) -> str:
        try:
            messages = [{"role": "system", "content": self.config.system_message}] + self.conversation_history
            
            if self.config.use_local:
                response = self._get_local_response(messages)
            else:
                response = self._get_cloud_response(messages)
                
            return response
        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _get_local_response(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.config.local_endpoint}/chat/completions"
        payload = {
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _get_cloud_response(self, messages: List[Dict[str, str]]) -> str:
        response = self.config.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

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
    finally:
        os.unlink(temp_path)

def speak(text):
    print("Speaking:", text)
    tts = gTTS(text=text, lang='en')
    tts.save("/tmp/reply.mp3")
    os.system("mpg123 /tmp/reply.mp3")

def main():
    # Configure LLM settings
    config = LLMConfig(
        use_local=True,  # Set to False to use OpenAI cloud
        local_endpoint="http://172.20.10.3:1234/v1",
        cloud_api_key=os.getenv("OPENAI_API_KEY"),
        system_message="You are RoverByte R1, a helpful AI assistant. Keep responses concise and conversational.",
        max_tokens=150
    )
    
    conversation_manager = ConversationManager(config)
    print("Assistant ready. Hold the button and speak.")
    
    try:
        while True:
            if GPIO.input(BUTTON) == GPIO.LOW:
                print("Button pressed!")
                time.sleep(0.1)  # Debounce
                if GPIO.input(BUTTON) == GPIO.LOW:
                    # Visual feedback - red LED while processing
                    GPIO.output(LED_GREEN, GPIO.LOW)
                    GPIO.output(LED_RED, GPIO.HIGH)
                    
                    user_text = record_audio()
                    if user_text:
                        conversation_manager.add_user_message(user_text)
                        response = conversation_manager.get_llm_response()
                        conversation_manager.add_assistant_message(response)
                        speak(response)
                    
                    # Reset LED state
                    GPIO.output(LED_RED, GPIO.LOW)
                    GPIO.output(LED_GREEN, GPIO.HIGH)
                    
                time.sleep(0.5)  # Delay before allowing next press

    except KeyboardInterrupt:
        print("\nExiting...")
        GPIO.cleanup()

if __name__ == "__main__":
    main()
