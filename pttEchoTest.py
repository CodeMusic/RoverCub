#!/usr/bin/env python3
"""
pttEcho_rgb_animated.py with button-triggered animations.

- Button on GPIO16 → manual_red animation
- Button on GPIO20 → manual_green animation
- Button on GPIO21 → manual_blue animation
"""

import os
import time
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
import random
import pyaudio

# Global variables
client = None
SYSTEM_MESSAGE = """You are a helpful assistant who knows psychology and computer science. Respond in only a sentence or two."""

# Sound Constants
SAMPLE_RATE = 44100
DURATION = 0.1  # seconds
VOLUME = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description='Push-to-talk echo with LED feedback')
    parser.add_argument('--test', action='store_true', help='Run audio test')
    parser.add_argument('--text', action='store_true', help='Use text input instead of voice')
    parser.add_argument('--device', default='plughw:1,0', help='ALSA device')
    parser.add_argument('--rate', type=int, default=48000, help='Sample rate')
    parser.add_argument('--period', type=int, default=128, help='Period size')
    return parser.parse_args()

def setup_openai():
    global client
    # Hardcoded API key for now
    API_KEY = ""
    client = openai.OpenAI(api_key=API_KEY)
    print("[INFO] Using OpenAI API")

def get_ai_response(text, use_openai=True):
    if use_openai:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
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
        return response.json()["choices"][0]["message"]["content"]

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
    os.remove(file_path)  # Clean up the file after playing

def test_audio(device, rate, period):
    print("[TEST] Starting audio test...")
    print("\n[AUDIO] Available playback devices:")
    subprocess.run(["aplay", "-l"], check=False)
    print("\n[AUDIO] Available recording devices:")
    subprocess.run(["arecord", "-l"], check=False)
    print("\n[TEST] Testing recording for 2 seconds...")
    test_wav = "/dev/shm/test_rec.wav"
    rec_cmd = ['arecord', '-D', device, '-r', str(rate), '-c', '1', '-f', 'S16_LE',
               '--period-size', str(period), '--buffer-size', str(period * 4), '-d', '2', test_wav]
    subprocess.run(rec_cmd, check=False)
    print("\n[TEST] Testing playback...")
    play_cmd = ['aplay', '-D', device, '-r', str(rate), '--period-size', str(period), '-q', test_wav]
    subprocess.run(play_cmd, check=False)
    if os.path.exists(test_wav):
        os.remove(test_wav)
    print("\n[TEST] Audio test complete")

def generate_tone(frequency, duration=DURATION):
    """Generate a simple sine wave tone"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * VOLUME
    return tone.astype(np.float32)

def play_tone(tone):
    """Play a tone using pyaudio"""
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
    """Play a startup jingle"""
    notes = [440, 523, 587, 659]  # A4, C5, D5, E5
    for note in notes:
        tone = generate_tone(note, 0.1)
        play_tone(tone)
        time.sleep(0.05)

def play_button_sound(button):
    """Play different sounds for each button"""
    frequencies = {
        'red': 440,    # A4
        'green': 523,  # C5
        'blue': 587    # D5
    }
    tone = generate_tone(frequencies[button])
    play_tone(tone)

def play_event_sound(event):
    """Play sounds for game events"""
    if event == 'win':
        # Play ascending scale
        for note in [440, 494, 523, 587, 659]:  # A4, B4, C5, D5, E5
            tone = generate_tone(note, 0.1)
            play_tone(tone)
            time.sleep(0.05)
    elif event == 'lose':
        # Play descending scale
        for note in [659, 587, 523, 494, 440]:  # E5, D5, C5, B4, A4
            tone = generate_tone(note, 0.1)
            play_tone(tone)
            time.sleep(0.05)
    elif event == 'level_up':
        # Play two ascending notes
        for note in [523, 587]:  # C5, D5
            tone = generate_tone(note, 0.1)
            play_tone(tone)
            time.sleep(0.05)

def main():
    args = parse_args()
    if args.test:
        test_audio(args.device, args.rate, args.period)
        return

    # Game Constants
    GRID_WIDTH = 8
    GRID_HEIGHT = 4
    TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT
    TICK_DURATION = 0.1  # Reduced from 0.5 to 0.1 for faster updates
    COOLDOWN_TICKS = 10
    WIN_TICKS = 100
    LOSE_MONOCHROME_TICKS = 10
    LOSE_STAGNANT_TICKS = 15
    MONOCHROME_THRESHOLD = 0.8  # 80% of one color

    # Colors
    COLORS = {
        'red': (64, 0, 0),
        'green': (0, 64, 0),
        'blue': (0, 0, 64),
        'black': (0, 0, 0)
    }

    # Game state
    current_state = 'normal'
    game_state = {
        'grid': [[random.choice(['red', 'green', 'blue']) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)],
        'score': 0,
        'level': 1,
        'ticks': 0,
        'last_button_press': None,
        'button_sequence': [],
        'cooldowns': {'red': 0, 'green': 0, 'blue': 0},
        'monochrome_count': 0,
        'stagnant_count': 0,
        'last_grid_hash': None
    }

    # LED setup
    LED_COUNT   = 32
    LED_PIN     = 18
    LED_FREQ_HZ = 800000
    LED_DMA     = 10
    LED_BRIGHT  = 32
    LED_INVERT  = False
    LED_CHANNEL = 0

    GREEN = (0, 64, 0)
    BLUE  = (0, 0, 64)
    RED   = (64, 0, 0)

    LEDS_OK = False
    try:
        from rpi_ws281x import PixelStrip, Color
        strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                          LED_INVERT, LED_BRIGHT, LED_CHANNEL)
        strip.begin()
        LEDS_OK = True
        print("[LED] initialized")
    except Exception as e:
        print("[LED] init failed:", e)

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

    # Startup animation
    if LEDS_OK:
        for step in range(LED_COUNT+1):
            for i in range(strip.numPixels()):
                if i < step:
                    strip.setPixelColor(i, Color(*BLUE))
                else:
                    strip.setPixelColor(i, Color(0,0,0))
            strip.show()
            time.sleep(0.01)  # Reduced from 1.0/LED_COUNT to 0.01 for faster animation
        time.sleep(0.1)  # Reduced from 0.2 to 0.1

    # Prompt user for API choice
    print("\n[CONFIG] Please choose your AI provider:")
    print("1. OpenAI (requires API key)")
    print("2. LM Studio (local)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    while choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")
        choice = input("Enter your choice (1 or 2): ").strip()

    # Always set up OpenAI client for speech-to-text
    setup_openai()
    
    # Set up LM Studio if chosen
    if choice == '2':
        print("[INFO] Using LM Studio at http://10.0.0.105:1234")

    if args.text:
        print("\n[INFO] Running in text mode. Type your message and press Enter.")
        print("Type 'quit' to exit.")
        while True:
            user_input = input("\nYour message: ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue
                
            print("\n[AI] Getting response...")
            ai_response = get_ai_response(user_input, choice == '1')
            print(f"[AI] Response: {ai_response}")
            
            tts_file = text_to_speech(ai_response)
            play_audio(tts_file)
        return

    # GPIO button setup
    RED_BTN = 20
    GRN_BTN = 16
    BLU_BTN = 26
    GPIO.setmode(GPIO.BCM)
    for btn in [RED_BTN, GRN_BTN, BLU_BTN]:
        GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    dev = InputDevice(EVENT_PATH)
    print("Listening on", dev.path)

    state = 'idle'
    idle_offset = 0
    breath_phase = 0.0
    last_tick = time.monotonic()

    while True:
        # Handle button events
        ev = dev.read_one()
        if ev and ev.type == ecodes.EV_KEY:
            key = categorize(ev)
            if key.scancode == ecodes.KEY_PLAYPAUSE:
                if key.keystate == key.key_down:
                    if current_state == 'normal':
                        current_state = handle_normal_button('siri')
                    else:
                        current_state = handle_game_button('siri')

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
        if current_state == 'game' and now - last_tick >= TICK_DURATION:
            update_grid()
            game_state['ticks'] += 1
            last_tick = now

        # Update LEDs
        update_leds_for_state()

        # Small sleep to prevent CPU hogging
        time.sleep(0.01)

    def get_neighbors(x, y):
        """Get the colors of neighboring cells (up, down, left, right)"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                neighbors.append(game_state['grid'][ny][nx])
        return neighbors

    def update_cell(x, y):
        """Update a cell based on its neighbors"""
        cell_color = game_state['grid'][y][x]
        neighbors = get_neighbors(x, y)
        
        # Count color occurrences
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        for color in neighbors:
            color_counts[color] += 1
        
        # Apply rules
        if color_counts[cell_color] >= 2:
            return cell_color  # Stabilize
        elif max(color_counts.values()) >= 3:
            return max(color_counts.items(), key=lambda x: x[1])[0]  # Mutate to majority
        else:
            # Rotate colors
            color_order = ['red', 'green', 'blue']
            current_index = color_order.index(cell_color)
            return color_order[(current_index + 1) % 3]

    def update_grid():
        """Update the entire grid for one tick"""
        new_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                new_grid[y][x] = update_cell(x, y)
        game_state['grid'] = new_grid

    def check_game_state():
        """Check win/lose conditions"""
        # Count colors
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        for row in game_state['grid']:
            for color in row:
                color_counts[color] += 1
        
        # Check for monochrome
        max_count = max(color_counts.values())
        if max_count >= TOTAL_CELLS * MONOCHROME_THRESHOLD:
            game_state['monochrome_count'] += 1
            if game_state['monochrome_count'] >= LOSE_MONOCHROME_TICKS:
                play_event_sound('lose')
                return 'lose_monochrome'
        else:
            game_state['monochrome_count'] = 0
        
        # Check for stagnation
        current_hash = hash(str(game_state['grid']))
        if current_hash == game_state['last_grid_hash']:
            game_state['stagnant_count'] += 1
            if game_state['stagnant_count'] >= LOSE_STAGNANT_TICKS:
                play_event_sound('lose')
                return 'lose_stagnant'
        else:
            game_state['stagnant_count'] = 0
        game_state['last_grid_hash'] = current_hash
        
        # Check for win
        if game_state['ticks'] >= WIN_TICKS:
            play_event_sound('win')
            return 'win'
        
        # Check for level up
        if game_state['ticks'] % 20 == 0 and game_state['ticks'] > 0:
            play_event_sound('level_up')
            game_state['level'] += 1
        
        return 'playing'

    def handle_game_button(button):
        """Handle button presses in game mode"""
        if button == 'siri':
            return 'normal'  # Return to normal mode
        
        # Check cooldown
        if game_state['cooldowns'][button] > 0:
            print(f"[GAME] {button} button on cooldown for {game_state['cooldowns'][button]} more ticks")
            return 'game'
        
        # Play button sound
        play_button_sound(button)
        
        # Apply button effects
        if button == 'red':
            # Inject red into 3 random cells
            for _ in range(3):
                x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
                game_state['grid'][y][x] = 'red'
        elif button == 'green':
            # Balance nearest area (center 2x2)
            center_x, center_y = GRID_WIDTH//2, GRID_HEIGHT//2
            for y in range(center_y-1, center_y+1):
                for x in range(center_x-1, center_x+1):
                    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                        game_state['grid'][y][x] = random.choice(['red', 'green', 'blue'])
        elif button == 'blue':
            # Freeze state for 3 ticks
            game_state['ticks'] = max(0, game_state['ticks'] - 3)
        
        # Set cooldown
        game_state['cooldowns'][button] = COOLDOWN_TICKS
        return 'game'

    def update_leds_for_state():
        """Update LED display based on current state"""
        if current_state == 'normal':
            if LEDS_OK:
                b = 0.5 + 0.5 * math.sin(breath_phase)
                set_brightness(LED_BRIGHT * b)
                for i in range(strip.numPixels()):
                    if (i + idle_offset) % 4 == 0:
                        strip.setPixelColor(i, Color(*BLUE))
                    else:
                        strip.setPixelColor(i, Color(0,0,0))
                strip.show()
        elif current_state == 'game':
            if LEDS_OK:
                # Display game grid
                for y in range(GRID_HEIGHT):
                    for x in range(GRID_WIDTH):
                        idx = y * GRID_WIDTH + x
                        color = COLORS[game_state['grid'][y][x]]
                        strip.setPixelColor(idx, Color(*color))
                strip.show()

    def handle_normal_button(button):
        """Handle button presses in normal mode"""
        if button == 'siri':
            return 'normal'
        else:
            # Play startup jingle when entering game mode
            play_jingle()
            print("[GAME] Entering game mode")
            game_state['last_button_press'] = button
            game_state['button_sequence'] = [button]
            return 'game'

    # Initialize game
    initialize_grid()

    EVENT_PATH = '/dev/input/event0'
    CARD_DEV   = args.device
    RATE       = str(args.rate)
    PERIOD     = str(args.period)
    BUFFER     = str(int(PERIOD)*4)
    TMP_WAV    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptt_clip.wav')
    RECORD_SEC = 3  # 3 seconds recording

    # Game mode states
    GAME_STATES = {
        'normal': 'Normal mode - Press any button to enter game mode',
        'game': 'Game mode - Press Siri button to return to normal mode'
    }

    # Initialize state
    current_state = 'normal'
    game_state = {
        'score': 0,
        'level': 1,
        'last_button_press': None,
        'button_sequence': []
    }

    def initialize_grid():
        """Initialize the grid with random colors"""
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                game_state['grid'][y][x] = random.choice(['red', 'green', 'blue'])

    def get_neighbors(x, y):
        """Get the colors of neighboring cells (up, down, left, right)"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                neighbors.append(game_state['grid'][ny][nx])
        return neighbors

    def update_cell(x, y):
        """Update a cell based on its neighbors"""
        cell_color = game_state['grid'][y][x]
        neighbors = get_neighbors(x, y)
        
        # Count color occurrences
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        for color in neighbors:
            color_counts[color] += 1
        
        # Apply rules
        if color_counts[cell_color] >= 2:
            return cell_color  # Stabilize
        elif max(color_counts.values()) >= 3:
            return max(color_counts.items(), key=lambda x: x[1])[0]  # Mutate to majority
        else:
            # Rotate colors
            color_order = ['red', 'green', 'blue']
            current_index = color_order.index(cell_color)
            return color_order[(current_index + 1) % 3]

    def update_grid():
        """Update the entire grid for one tick"""
        new_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                new_grid[y][x] = update_cell(x, y)
        game_state['grid'] = new_grid

    def check_game_state():
        """Check win/lose conditions"""
        # Count colors
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        for row in game_state['grid']:
            for color in row:
                color_counts[color] += 1
        
        # Check for monochrome
        max_count = max(color_counts.values())
        if max_count >= TOTAL_CELLS * MONOCHROME_THRESHOLD:
            game_state['monochrome_count'] += 1
            if game_state['monochrome_count'] >= LOSE_MONOCHROME_TICKS:
                play_event_sound('lose')
                return 'lose_monochrome'
        else:
            game_state['monochrome_count'] = 0
        
        # Check for stagnation
        current_hash = hash(str(game_state['grid']))
        if current_hash == game_state['last_grid_hash']:
            game_state['stagnant_count'] += 1
            if game_state['stagnant_count'] >= LOSE_STAGNANT_TICKS:
                play_event_sound('lose')
                return 'lose_stagnant'
        else:
            game_state['stagnant_count'] = 0
        game_state['last_grid_hash'] = current_hash
        
        # Check for win
        if game_state['ticks'] >= WIN_TICKS:
            play_event_sound('win')
            return 'win'
        
        # Check for level up
        if game_state['ticks'] % 20 == 0 and game_state['ticks'] > 0:
            play_event_sound('level_up')
            game_state['level'] += 1
        
        return 'playing'

    def handle_game_button(button):
        """Handle button presses in game mode"""
        if button == 'siri':
            return 'normal'  # Return to normal mode
        
        # Check cooldown
        if game_state['cooldowns'][button] > 0:
            print(f"[GAME] {button} button on cooldown for {game_state['cooldowns'][button]} more ticks")
            return 'game'
        
        # Play button sound
        play_button_sound(button)
        
        # Apply button effects
        if button == 'red':
            # Inject red into 3 random cells
            for _ in range(3):
                x, y = random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)
                game_state['grid'][y][x] = 'red'
        elif button == 'green':
            # Balance nearest area (center 2x2)
            center_x, center_y = GRID_WIDTH//2, GRID_HEIGHT//2
            for y in range(center_y-1, center_y+1):
                for x in range(center_x-1, center_x+1):
                    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                        game_state['grid'][y][x] = random.choice(['red', 'green', 'blue'])
        elif button == 'blue':
            # Freeze state for 3 ticks
            game_state['ticks'] = max(0, game_state['ticks'] - 3)
        
        # Set cooldown
        game_state['cooldowns'][button] = COOLDOWN_TICKS
        return 'game'

    def update_leds_for_state():
        """Update LED display based on current state"""
        if current_state == 'normal':
            if LEDS_OK:
                b = 0.5 + 0.5 * math.sin(breath_phase)
                set_brightness(LED_BRIGHT * b)
                for i in range(strip.numPixels()):
                    if (i + idle_offset) % 4 == 0:
                        strip.setPixelColor(i, Color(*BLUE))
                    else:
                        strip.setPixelColor(i, Color(0,0,0))
                strip.show()
        elif current_state == 'game':
            if LEDS_OK:
                # Display game grid
                for y in range(GRID_HEIGHT):
                    for x in range(GRID_WIDTH):
                        idx = y * GRID_WIDTH + x
                        color = COLORS[game_state['grid'][y][x]]
                        strip.setPixelColor(idx, Color(*color))
                strip.show()

    # Initialize game
    initialize_grid()

    EVENT_PATH = '/dev/input/event0'
    CARD_DEV   = args.device
    RATE       = str(args.rate)
    PERIOD     = str(args.period)
    BUFFER     = str(int(PERIOD)*4)
    TMP_WAV    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptt_clip.wav')
    RECORD_SEC = 3  # 3 seconds recording

    # Game mode states
    GAME_STATES = {
        'normal': 'Normal mode - Press any button to enter game mode',
        'game': 'Game mode - Press Siri button to return to normal mode'
    }

    # Initialize state
    current_state = 'normal'
    game_state = {
        'score': 0,
        'level': 1,
        'last_button_press': None,
        'button_sequence': []
    }

    def handle_normal_button(button):
        """Handle button presses in normal mode"""
        if button == 'siri':
            return 'normal'
        else:
            # Play startup jingle when entering game mode
            play_jingle()
            print("[GAME] Entering game mode")
            game_state['last_button_press'] = button
            game_state['button_sequence'] = [button]
            return 'game'

    # GPIO button setup
    RED_BTN = 20
    GRN_BTN = 16
    BLU_BTN = 26
    GPIO.setmode(GPIO.BCM)
    for btn in [RED_BTN, GRN_BTN, BLU_BTN]:
        GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    dev = InputDevice(EVENT_PATH)
    print("Listening on", dev.path)

    state           = 'idle'
    idle_offset     = 0
    breath_phase    = 0.0
    rec_proc        = None
    play_proc       = None
    press_t         = 0.0
    processing_end  = 0.0
    speak_phase     = 0.0

    while True:
        # Handle button events
        ev = dev.read_one()
        if ev and ev.type == ecodes.EV_KEY:
            key = categorize(ev)
            if key.scancode == ecodes.KEY_PLAYPAUSE:
                if key.keystate == key.key_down:
                    press_t = time.monotonic()
                    if current_state == 'normal':
                        current_state = handle_normal_button('siri')
                    else:
                        current_state = handle_game_button('siri')
                elif key.keystate == key.key_up and state == 'recording':
                    duration = time.monotonic() - recording_start
                    print(f"[DEBUG] Button released after {duration:.3f} seconds")
                    if rec_proc:
                        print("[DEBUG] Waiting for recording to complete...")
                        try:
                            rec_proc.wait(timeout=RECORD_SEC + 1)
                        except subprocess.TimeoutExpired:
                            print("[WARNING] Recording did not complete in time")
                            rec_proc.kill()
                        rec_proc = None
                    
                    if not os.path.exists(TMP_WAV):
                        print("[DEBUG] Recording file not found")
                        state = 'idle'
                    else:
                        size_kb = os.path.getsize(TMP_WAV) / 1024
                        print(f"[PLAY] {duration:.2f}s | {size_kb:.1f} kB")
                        print(f"[DEBUG] Recording saved to: {TMP_WAV}")
                        
                        if current_state == 'normal':
                            # Only process audio in normal mode
                            print("[STT] Converting speech to text...")
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
                        
                        state = 'processing'
                        processing_end = time.monotonic() + 0.5

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

        # Update LEDs based on current state
        update_leds_for_state()

        # Handle other states
        now = time.monotonic()
        breath_phase += 0.05

        if state == 'idle':
            time.sleep(0.1)

        elif state == 'recording':
            time.sleep(0.05)

        elif state == 'processing':
            if now >= processing_end:
                state = 'idle'
            else:
                time.sleep(0.05)

        elif state == 'speaking':
            if play_proc and play_proc.poll() is not None:
                play_proc.wait()
                print("[DEBUG] Playback completed")
                set_brightness(LED_BRIGHT)
                state = 'idle'
            else:
                time.sleep(0.02)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        try:
            GPIO.cleanup()
        except:
            pass
        if 'LEDS_OK' in globals() and LEDS_OK:
            show_color((0,0,0))
        if 'rec_proc' in globals() and rec_proc:
            rec_proc.terminate()
        if 'play_proc' in globals() and play_proc:
            play_proc.terminate()
        if os.path.exists('/dev/shm/ptt_clip.wav'):
            os.remove('/dev/shm/ptt_clip.wav')
        print("Clean exit complete.")