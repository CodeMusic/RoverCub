#!/usr/bin/env python3
"""
RoverCub - Combined LLM Assistant and Reflex Grid Game

Features:
- Voice/Text input with LLM responses
- Reflex Grid Game with LED feedback
- Red, Green, Blue buttons for both modes
- Game mode: Balance colors to reach purple
- Normal mode: Interact with LLM
- Token visualization on LED grid
"""

import os
import time
import random
import subprocess
import math
import argparse
import signal
import RPi.GPIO as GPIO
from evdev import InputDevice, categorize, ecodes
import sounddevice as sd
import numpy as np
import soundfile as sf
import openai
import requests
import pyaudio
import tiktoken
from rpi_ws281x import PixelStrip, Color

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/GPT-4 encoding

def tokenize_offline(text, max_tokens=32):
    return enc.encode(text)[:max_tokens]

# Global variables
client = None
SYSTEM_MESSAGE = """You are a helpful assistant who knows psychology and computer science. Respond in only a sentence or two."""

# Sound Constants
SAMPLE_RATE = 44100
DURATION = 0.1  # seconds
VOLUME = 0.5

# LED Configuration
GRID_WIDTH = 8
GRID_HEIGHT = 4
LED_COUNT = GRID_WIDTH * GRID_HEIGHT
LED_PIN = 18
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 32
LED_INVERT = False
LED_CHANNEL = 0

# LED State
LEDS_OK = False
strip = None

# GPIO Button Pins
RED_BTN = 20
GRN_BTN = 16
BLU_BTN = 26

# Game Timing
INITIAL_TICK = 0.5
MIN_TICK = 0.1
MAX_TICK = 2.0

# Color Mapping - Extended for token visualization
COLORS = {
    'red': (64, 0, 0),
    'green': (0, 64, 0),
    'blue': (0, 0, 64),
    'purple': (32, 0, 32),
    'black': (0, 0, 0),
    # Additional colors for token visualization
    'yellow': (64, 64, 0),
    'cyan': (0, 64, 64),
    'magenta': (64, 0, 64),
    'orange': (64, 32, 0),
    'pink': (64, 0, 32),
    'teal': (0, 32, 32),
    'lime': (32, 64, 0),
    'indigo': (32, 0, 64),
    'maroon': (32, 0, 0),
    'olive': (32, 32, 0),
    'navy': (0, 0, 32),
    'aqua': (0, 64, 32),
    'fuchsia': (64, 0, 64),
    'silver': (48, 48, 48),
    'gray': (32, 32, 32),
    'white': (64, 64, 64)
}

def parse_args():
    parser = argparse.ArgumentParser(description='RoverCub - Combined LLM Assistant and Reflex Grid Game')
    parser.add_argument('--test', action='store_true', help='Run audio test')
    parser.add_argument('--text', action='store_true', help='Use text input instead of voice')
    parser.add_argument('--device', default='plughw:1,0', help='ALSA device')
    parser.add_argument('--rate', type=int, default=48000, help='Sample rate')
    parser.add_argument('--period', type=int, default=128, help='Period size')
    return parser.parse_args()

def setup_openai():
    global client
    API_KEY = ""
    client = openai.OpenAI(api_key=API_KEY)
    print("[INFO] Using OpenAI API")

def visualize_tokens(tokens):
    """Visualize tokens on the LED grid using a variety of colors"""
    global LEDS_OK, strip
    if not LEDS_OK:
        return
    
    # Get all available colors except black
    available_colors = list(COLORS.keys())
    available_colors.remove('black')
    
    # Ensure we have exactly 32 tokens (pad with 0 if needed)
    padded_tokens = tokens[:32]  # Take first 32 tokens
    if len(padded_tokens) < 32:
        padded_tokens.extend([0] * (32 - len(padded_tokens)))  # Pad with zeros
    
    # Map each token to a color based on its value
    for i, token in enumerate(padded_tokens):
        if token == 0:  # Space or padding
            strip.setPixelColor(i, Color(0, 0, 0))  # Black for spaces
        else:
            # Use token value to select a color
            color_name = available_colors[token % len(available_colors)]
            strip.setPixelColor(i, Color(*COLORS[color_name]))
    
    strip.show()
    print(f"[LED] Visualized {len(tokens)} tokens on {LED_COUNT} LEDs")

def get_ai_response(text, use_openai=True):
    if use_openai:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": text}
            ]
        )
        response_text = response.choices[0].message.content
        # Visualize tokens
        tokens = tokenize_offline(response_text)
        print(f"[TOKENS] Generated {len(tokens)} tokens")
        visualize_tokens(tokens)
        return response_text
    else:
        response = requests.post(
            "http://10.0.0.105:1234/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        )
        response_text = response.json()["choices"][0]["message"]["content"]
        # Visualize tokens
        tokens = tokenize_offline(response_text)
        print(f"[TOKENS] Generated {len(tokens)} tokens")
        visualize_tokens(tokens)
        return response_text

def text_to_speech(text):
    print("[TTS] Converting response to speech...")
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    tts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tts_response.mp3')
    response.stream_to_file(tts_file)
    return tts_file

def play_audio(file_path):
    print("[PLAY] Playing audio response...")
    play_cmd = ['mpg123', '-q', file_path]
    subprocess.run(play_cmd)
    os.remove(file_path)

def generate_tone(frequency, duration=DURATION):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * VOLUME
    return tone.astype(np.float32)

def play_tone(tone):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True)
    stream.write(tone)
    stream.stop_stream()
    stream.close()
    p.terminate()

def play_jingle():
    notes = [440, 523, 587, 659]  # A4, C5, D5, E5
    for note in notes:
        tone = generate_tone(note, 0.1)
        play_tone(tone)
        time.sleep(0.05)

def play_button_sound(button):
    frequencies = {
        'red': 440,    # A4
        'green': 523,  # C5
        'blue': 587    # D5
    }
    tone = generate_tone(frequencies[button])
    play_tone(tone)

def play_event_sound(event):
    if event == 'win':
        for note in [440, 494, 523, 587, 659]:  # A4, B4, C5, D5, E5
            tone = generate_tone(note, 0.1)
            play_tone(tone)
            time.sleep(0.05)
    elif event == 'lose':
        for note in [659, 587, 523, 494, 440]:  # E5, D5, C5, B4, A4
            tone = generate_tone(note, 0.1)
            play_tone(tone)
            time.sleep(0.05)
    elif event == 'level_up':
        for note in [523, 587]:  # C5, D5
            tone = generate_tone(note, 0.1)
            play_tone(tone)
            time.sleep(0.05)

def test_audio(device, rate, period):
    """Test audio input and output functionality"""
    print("\n[TEST] Starting audio test...")
    print(f"[TEST] Using device: {device}")
    print(f"[TEST] Sample rate: {rate}")
    print(f"[TEST] Period size: {period}")
    
    # Test recording
    print("\n[TEST] Testing recording...")
    test_wav = "test_recording.wav"
    try:
        rec_cmd = ['arecord', '-D', device, '-r', str(rate), '-f', 'S16_LE', '-c', '1', '-t', 'wav', test_wav]
        print(f"[TEST] Running command: {' '.join(rec_cmd)}")
        rec_proc = subprocess.Popen(rec_cmd)
        print("[TEST] Recording for 3 seconds...")
        time.sleep(3)
        rec_proc.terminate()
        print("[TEST] Recording stopped")
        
        if os.path.exists(test_wav):
            print("[TEST] Recording successful")
            # Test playback
            print("\n[TEST] Testing playback...")
            play_cmd = ['aplay', '-D', device, test_wav]
            print(f"[TEST] Running command: {' '.join(play_cmd)}")
            subprocess.run(play_cmd)
            print("[TEST] Playback complete")
            
            # Clean up
            os.remove(test_wav)
        else:
            print("[TEST] Recording failed - no file created")
    except Exception as e:
        print(f"[TEST] Error during audio test: {e}")
    
    # Test button detection
    print("\n[TEST] Testing button detection...")
    try:
        for event_file in os.listdir('/dev/input/by-id'):
            if 'usb-Walmart_AB13X_Headset_Adapter' in event_file:
                event_path = os.path.join('/dev/input/by-id', event_file)
                dev = InputDevice(event_path)
                print(f"[TEST] Found headset at {dev.path}")
                print("[TEST] Press and release the button to test...")
                
                button_pressed = False
                start_time = time.time()
                while time.time() - start_time < 10:  # Test for 10 seconds
                    ev = dev.read_one()
                    if ev and ev.type == ecodes.EV_KEY:
                        key = categorize(ev)
                        if key.scancode == ecodes.KEY_PLAYPAUSE:
                            if key.keystate == key.key_down and not button_pressed:
                                button_pressed = True
                                print("[TEST] Button pressed")
                            elif key.keystate == key.key_up and button_pressed:
                                button_pressed = False
                                print("[TEST] Button released")
                    time.sleep(0.01)
                break
    except Exception as e:
        print(f"[TEST] Error during button test: {e}")
    
    print("\n[TEST] Audio test complete")

def main():
    global LEDS_OK, strip
    args = parse_args()
    if args.test:
        test_audio(args.device, args.rate, args.period)
        return

    # Initialize LED Strip
    try:
        strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                          LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        strip.begin()
        LEDS_OK = True
        print("[LED] initialized")
    except Exception as e:
        print("[LED] init failed:", e)
        LEDS_OK = False

    def show_color(color):
        if not LEDS_OK:
            return
        c = Color(color[0], color[1], color[2])
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, c)
        strip.show()

    def set_brightness(val):
        if not LEDS_OK:
            return
        strip.setBrightness(max(0, min(255, int(val))))
        strip.show()

    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    for btn in [RED_BTN, GRN_BTN, BLU_BTN]:
        GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Game state
    grid = [[random.choice(['red', 'green', 'blue']) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    tick_duration = INITIAL_TICK
    purple_phase = False
    purple_phase_stage = 0
    current_state = 'normal'  # 'normal' or 'game'
    last_tick = time.monotonic()
    last_correct_guess = 0  # Time of last correct guess
    guess_cooldown = 0.5  # Cooldown period in seconds

    def draw_grid():
        if not LEDS_OK:
            return
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                idx = y * GRID_WIDTH + x
                color = COLORS[grid[y][x]]
                strip.setPixelColor(idx, Color(*color))
        strip.show()

    def get_dominant_color():
        counts = {'red': 0, 'green': 0, 'blue': 0, 'purple': 0}
        for row in grid:
            for color in row:
                if color in counts:
                    counts[color] += 1
        return max(counts, key=counts.get), counts

    def add_random_color():
        x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
        grid[y][x] = random.choice(['red', 'green', 'blue'])

    def convert_cells_to_purple(count=4):
        converted = 0
        while converted < count:
            x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            if grid[y][x] in ['red', 'green', 'blue']:
                grid[y][x] = 'purple'
                converted += 1

    def get_button_press(timeout):
        start = time.time()
        while time.time() - start < timeout:
            if GPIO.input(RED_BTN) == GPIO.LOW:
                return 'red'
            elif GPIO.input(GRN_BTN) == GPIO.LOW:
                return 'green'
            elif GPIO.input(BLU_BTN) == GPIO.LOW:
                return 'blue'
            time.sleep(0.01)
        return None

    def handle_game_button(button):
        nonlocal purple_phase, purple_phase_stage, tick_duration, current_state, last_correct_guess
        
        if button == 'siri':
            return 'normal'
        
        # Check if we're in cooldown period
        now = time.monotonic()
        if now - last_correct_guess < guess_cooldown:
            return 'game'
        
        if not purple_phase:
            dominant, counts = get_dominant_color()
            if button == dominant:
                convert_cells_to_purple()
                tick_duration = max(MIN_TICK, tick_duration - 0.05)
                last_correct_guess = now  # Update last correct guess time
                print(f"[HIT] Correct ({button}), speed up -> {tick_duration:.2f}s")
            else:
                for _ in range(4):
                    add_random_color()
                tick_duration = min(MAX_TICK, tick_duration + 0.1)
                print(f"[MISS] Wrong ({button}), slowed -> {tick_duration:.2f}s")
        else:
            if purple_phase_stage == 0 and button == 'red':
                purple_phase_stage = 1
                print("[PHASE] Red pressed. Now press blue.")
            elif purple_phase_stage == 1 and button == 'blue':
                print("[RESET] Grid resetting...")
                grid = [[random.choice(['red', 'green', 'blue']) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
                tick_duration = INITIAL_TICK
                purple_phase = False
                purple_phase_stage = 0
        
        return 'game'

    def handle_normal_button(button):
        nonlocal current_state
        if button == 'siri':
            return 'normal'
        else:
            play_jingle()
            print("[GAME] Entering game mode")
            return 'game'

    # Prompt user for API choice
    print("\n[CONFIG] Please choose your AI provider:")
    print("1. OpenAI (requires API key)")
    print("2. LM Studio (local)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    while choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")
        choice = input("Enter your choice (1 or 2): ").strip()

    setup_openai()
    
    if choice == '2':
        print("[INFO] Using LM Studio at http://10.0.0.105:1234")

    if args.text:
        print("\n[INFO] Running in text mode. Type your message and press Enter.")
        print("Type 'game' to enter game mode, 'quit' to exit.")
        while True:
            user_input = input("\nYour message: ").strip()
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'game':
                current_state = 'game'
                print("[GAME] Entering game mode")
                continue
                
            if current_state == 'normal':
                print("\n[AI] Getting response...")
                ai_response = get_ai_response(user_input, choice == '1')
                print(f"[AI] Response: {ai_response}")
                
                tts_file = text_to_speech(ai_response)
                play_audio(tts_file)
            else:
                # Handle game commands in text mode
                if user_input in ['red', 'green', 'blue', 'siri']:
                    current_state = handle_game_button(user_input)
                    if current_state == 'normal':
                        print("[GAME] Returning to normal mode")
                else:
                    print("[GAME] Invalid command. Use red, green, blue, or siri to exit")
            
            # Update game state
            if current_state == 'game':
                now = time.monotonic()
                if now - last_tick >= tick_duration:
                    add_random_color()
                    draw_grid()
                    
                    dominant, counts = get_dominant_color()
                    total_cells = GRID_WIDTH * GRID_HEIGHT
                    
                    if not purple_phase and counts['purple'] > total_cells * 0.6:
                        purple_phase = True
                        print("[PHASE] Purple dominates. Press red then blue to reset.")
                    
                    last_tick = now

    else:
        # Voice mode
        EVENT_PATH = '/dev/input/event0'
        CARD_DEV = args.device
        RATE = str(args.rate)
        PERIOD = str(args.period)
        BUFFER = str(int(PERIOD)*4)
        TMP_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptt_clip.wav')
        RECORD_SEC = 3

        # Ensure the temporary directory exists and is writable
        try:
            os.makedirs(os.path.dirname(TMP_WAV), exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Could not create temp directory: {e}")
            TMP_WAV = '/tmp/ptt_clip.wav'  # Fallback to /tmp
            print(f"[INFO] Using fallback path: {TMP_WAV}")

        # Find the correct event device for the headset
        try:
            for event_file in os.listdir('/dev/input/by-id'):
                if 'usb-Walmart_AB13X_Headset_Adapter' in event_file:
                    EVENT_PATH = os.path.join('/dev/input/by-id', event_file)
                    break
            dev = InputDevice(EVENT_PATH)
            print(f"[MIC] Found headset at {dev.path}")
        except Exception as e:
            print(f"[ERROR] Could not open headset device: {e}")
            print("Falling back to GPIO buttons only")
            dev = None

        state = 'idle'
        idle_offset = 0
        breath_phase = 0.0
        rec_proc = None
        play_proc = None
        press_t = 0.0
        processing_end = 0.0
        speak_phase = 0.0
        recording_start = 0.0
        button_pressed = False  # Track button state

        try:
            while True:
                # Handle button events
                if dev:
                    try:
                        ev = dev.read_one()
                        if ev and ev.type == ecodes.EV_KEY:
                            key = categorize(ev)
                            if key.scancode == ecodes.KEY_PLAYPAUSE:
                                if key.keystate == key.key_down and not button_pressed:
                                    button_pressed = True
                                    press_t = time.monotonic()
                                    recording_start = press_t
                                    print("[BUTTON] Button pressed")
                                    play_jingle()  # Play jingle when button is pressed
                                    
                                    if current_state == 'normal':
                                        current_state = handle_normal_button('siri')
                                        print("[STATE] Switched to normal mode")
                                    else:
                                        current_state = handle_game_button('siri')
                                        print("[STATE] Switched to game mode")
                                    
                                    # Start recording
                                    if current_state == 'normal':
                                        print("[REC] Starting recording...")
                                        try:
                                            rec_cmd = ['arecord', '-D', CARD_DEV, '-r', RATE, '-f', 'S16_LE', '-c', '1', '-t', 'wav', TMP_WAV]
                                            print(f"[REC] Running command: {' '.join(rec_cmd)}")
                                            rec_proc = subprocess.Popen(rec_cmd)
                                            state = 'recording'
                                        except Exception as e:
                                            print(f"[ERROR] Failed to start recording: {e}")
                                            state = 'idle'
                                    
                                elif key.keystate == key.key_up and button_pressed:
                                    button_pressed = False
                                    print("[BUTTON] Button released")
                                    if state == 'recording':
                                        duration = time.monotonic() - recording_start
                                        print(f"[REC] Recording stopped after {duration:.2f} seconds")
                                        if rec_proc:
                                            try:
                                                rec_proc.wait(timeout=RECORD_SEC + 1)
                                            except subprocess.TimeoutExpired:
                                                rec_proc.kill()
                                            rec_proc = None
                                        
                                        if os.path.exists(TMP_WAV):
                                            if current_state == 'normal':
                                                print("[STT] Converting speech to text...")
                                                try:
                                                    with open(TMP_WAV, "rb") as audio_file:
                                                        transcript = client.audio.transcriptions.create(
                                                            model="whisper-1",
                                                            file=audio_file
                                                        )
                                                    print(f"[STT] User said: {transcript.text}")
                                                    
                                                    print("[AI] Getting response...")
                                                    ai_response = get_ai_response(transcript.text, choice == '1')
                                                    print(f"[AI] Response: {ai_response}")
                                                    
                                                    tts_file = text_to_speech(ai_response)
                                                    play_audio(tts_file)
                                                except Exception as e:
                                                    print(f"[ERROR] Failed to process audio: {e}")
                                            
                                            state = 'processing'
                                            processing_end = time.monotonic() + 0.5
                    except Exception as e:
                        print(f"[ERROR] Reading from device: {e}")
                        time.sleep(0.1)  # Prevent tight loop on error

                # Handle GPIO buttons
                if GPIO.input(RED_BTN) == GPIO.LOW:
                    if current_state == 'normal':
                        current_state = handle_normal_button('red')
                    else:
                        current_state = handle_game_button('red')
                elif GPIO.input(GRN_BTN) == GPIO.LOW:
                    if current_state == 'normal':
                        current_state = handle_normal_button('green')
                    else:
                        current_state = handle_game_button('green')
                elif GPIO.input(BLU_BTN) == GPIO.LOW:
                    if current_state == 'normal':
                        current_state = handle_normal_button('blue')
                    else:
                        current_state = handle_game_button('blue')

                # Update game state
                now = time.monotonic()
                if current_state == 'game' and now - last_tick >= tick_duration:
                    add_random_color()
                    draw_grid()
                    
                    dominant, counts = get_dominant_color()
                    total_cells = GRID_WIDTH * GRID_HEIGHT
                    
                    if not purple_phase and counts['purple'] > total_cells * 0.6:
                        purple_phase = True
                        print("[PHASE] Purple dominates. Press red then blue to reset.")
                    
                    last_tick = now

                # Update LEDs
                if current_state == 'normal':
                    if LEDS_OK:
                        b = 0.5 + 0.5 * math.sin(breath_phase)
                        set_brightness(LED_BRIGHTNESS * b)
                        for i in range(strip.numPixels()):
                            if (i + idle_offset) % 4 == 0:
                                strip.setPixelColor(i, Color(*COLORS['blue']))
                            else:
                                strip.setPixelColor(i, Color(0,0,0))
                        strip.show()
                else:
                    draw_grid()

                # Update animation phases
                breath_phase += 0.05
                idle_offset = (idle_offset + 1) % 4

                # Small sleep to prevent CPU hogging
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[EXIT] Cleaning up...")
            GPIO.cleanup()
            if LEDS_OK:
                for i in range(LED_COUNT):
                    strip.setPixelColor(i, Color(0,0,0))
                strip.show()
            if 'rec_proc' in globals() and rec_proc:
                rec_proc.terminate()
            if 'play_proc' in globals() and play_proc:
                play_proc.terminate()
            if os.path.exists(TMP_WAV):
                os.remove(TMP_WAV)
            print("[DONE] Goodbye!")

if __name__ == "__main__":
    main()
