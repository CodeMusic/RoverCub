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
import socket
import sys
import subprocess
import pyaudio

# Suppress GPIO warnings
GPIO.setwarnings(False)

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

class AudioManager:
    def __init__(self):
        self.mic = None
        self.recognizer = None
        self.audio_available = False
        self._initialize_audio()

    def _initialize_audio(self):
        """Initialize audio devices if available"""
        try:
            audio = pyaudio.PyAudio()
            device_count = audio.get_device_count()
            
            if device_count > 0:
                print(f"\nFound {device_count} audio device(s):")
                for i in range(device_count):
                    device_info = audio.get_device_info_by_index(i)
                    print(f"Device {i}: {device_info['name']}")
                    if device_info['maxInputChannels'] > 0:
                        print("  - Has input capability")
                
                # Try to find an input device
                for i in range(device_count):
                    device_info = audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        print(f"\nUsing audio input device: {device_info['name']}")
                        self.recognizer = sr.Recognizer()
                        self.mic = sr.Microphone(device_index=i)
                        self.audio_available = True
                        break
                        
            audio.terminate()
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.audio_available = False

    def record_audio(self):
        """Record audio if available, otherwise return None"""
        if not self.audio_available:
            print("\nAudio input not available. Running in text-only mode.")
            print("To enable voice input:")
            print("1. Connect a USB microphone")
            print("2. Run: sudo apt-get install alsa-utils python3-pyaudio")
            print("3. Reboot the Raspberry Pi")
            return None
            
        with self.mic as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = self.recognizer.listen(source)
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
            except Exception as e:
                print(f"Recording error: {e}")
                return None

def speak(text):
    """Speak text if audio output is available"""
    print("Speaking:", text)
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("/tmp/reply.mp3")
        os.system("mpg123 /tmp/reply.mp3")
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        print("Running in text-only mode")

def get_local_ip() -> str:
    try:
        # Try to get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "unknown"

def check_network_connectivity(host: str, port: int) -> bool:
    print(f"\nNetwork Diagnostics:")
    print(f"Local IP: {get_local_ip()}")
    print(f"Target Host: {host}")
    print(f"Target Port: {port}")
    
    # Check DNS resolution
    try:
        print("\nTesting DNS resolution...")
        ip = socket.gethostbyname(host)
        print(f"Host {host} resolves to {ip}")
    except socket.gaierror:
        print(f"Could not resolve host {host}")
        return False
    
    # Check routing
    try:
        print("\nChecking routing...")
        route_result = subprocess.run(['ip', 'route', 'get', host], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
        print(f"Route to {host}:")
        print(route_result.stdout.decode())
    except Exception as e:
        print(f"Route check error: {e}")
    
    # Check if host is reachable
    try:
        print("\nTesting host reachability...")
        ping_result = subprocess.run(['ping', '-c', '1', host], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        if ping_result.returncode == 0:
            print(f"Host {host} is reachable")
        else:
            print(f"Host {host} is not reachable")
            print(f"Ping error: {ping_result.stderr.decode()}")
            return False
            
        # Check port
        print("\nTesting port connectivity...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"Port {port} is open")
            return True
        else:
            print(f"Port {port} is closed")
            return False
            
    except Exception as e:
        print(f"Network check error: {e}")
        return False

class LLMConfig:
    def __init__(
        self,
        use_local: bool = True,
        local_endpoint: str = "http://172.20.10.8:1234/v1",
        cloud_api_key: Optional[str] = None,
        system_message: str = "You are RoverByte R1, a helpful AI assistant. Keep responses concise and conversational.",
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
            
    def check_connection(self) -> bool:
        if not self.use_local:
            return True
            
        try:
            # Extract host and port from endpoint
            host = self.local_endpoint.split("://")[1].split(":")[0]
            port = int(self.local_endpoint.split(":")[2].split("/")[0])
            
            # Run detailed network check
            if not check_network_connectivity(host, port):
                return False
                
            # Try to get models list
            print("\nTesting API endpoint...")
            url = f"{self.local_endpoint}/models"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("Successfully connected to LM Studio API")
                return True
            else:
                print(f"API test failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Connection check error: {e}")
            return False

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
        print(f"Attempting to connect to: {url}")
        
        payload = {
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response.text}")
            raise

    def _get_cloud_response(self, messages: List[Dict[str, str]]) -> str:
        response = self.config.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

def main():
    # Initialize audio manager
    audio_manager = AudioManager()
    
    # Configure LLM settings
    config = LLMConfig(
        use_local=True,
        local_endpoint="http://172.20.10.8:1234/v1",
        cloud_api_key=os.getenv("OPENAI_API_KEY"),
        system_message="You are RoverByte R1, a helpful AI assistant. Keep responses concise and conversational.",
        max_tokens=150
    )
    
    conversation_manager = ConversationManager(config)
    
    if audio_manager.audio_available:
        print("Assistant ready. Hold the button and speak.")
    else:
        print("Assistant ready in text-only mode. Press button to start conversation.")
    
    try:
        while True:
            if GPIO.input(BUTTON) == GPIO.LOW:
                print("Button pressed!")
                time.sleep(0.1)  # Debounce
                if GPIO.input(BUTTON) == GPIO.LOW:
                    # Visual feedback - red LED while processing
                    GPIO.output(LED_GREEN, GPIO.LOW)
                    GPIO.output(LED_RED, GPIO.HIGH)
                    
                    user_text = audio_manager.record_audio()
                    if user_text:
                        conversation_manager.add_user_message(user_text)
                        response = conversation_manager.get_llm_response()
                        conversation_manager.add_assistant_message(response)
                        speak(response)
                    else:
                        # In text-only mode, use a default message
                        user_text = "Hello"
                        conversation_manager.add_user_message(user_text)
                        response = conversation_manager.get_llm_response()
                        conversation_manager.add_assistant_message(response)
                        print(f"Assistant: {response}")
                    
                    # Reset LED state
                    GPIO.output(LED_RED, GPIO.LOW)
                    GPIO.output(LED_GREEN, GPIO.HIGH)
                    
                time.sleep(0.5)  # Delay before allowing next press

    except KeyboardInterrupt:
        print("\nExiting...")
        GPIO.cleanup()

if __name__ == "__main__":
    main()
