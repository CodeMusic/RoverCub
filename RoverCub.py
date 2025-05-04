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

def parse_args():
    parser = argparse.ArgumentParser(description='Push-to-talk echo with LED feedback')
    parser.add_argument('--test', action='store_true', help='Run audio test')
    parser.add_argument('--device', default='plughw:1,0', help='ALSA device')
    parser.add_argument('--rate', type=int, default=48000, help='Sample rate')
    parser.add_argument('--period', type=int, default=128, help='Period size')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI API instead of LM Studio')
    return parser.parse_args()

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

def main():
    args = parse_args()
    if args.test:
        test_audio(args.device, args.rate, args.period)
        return

    # System message for both OpenAI and LM Studio
    SYSTEM_MESSAGE = """You are a helpful assistant that provides concise, direct answers.
    Keep responses brief and to the point.
    Focus on the most relevant information.
    Avoid unnecessary explanations or context."""

    # Set up API configuration
    if args.use_openai:
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            print("[ERROR] OPENAI_API_KEY environment variable not set")
            return
        print("[INFO] Using OpenAI API")
    else:
        import requests
        LM_STUDIO_URL = "http://10.0.0.105:1234"
        print("[INFO] Using LM Studio at", LM_STUDIO_URL)

    def get_ai_response(text):
        if args.use_openai:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        else:
            response = requests.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
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

    EVENT_PATH = '/dev/input/event0'
    CARD_DEV   = args.device
    RATE       = str(args.rate)
    PERIOD     = str(args.period)
    BUFFER     = str(int(PERIOD)*4)
    TMP_WAV    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptt_clip.wav')
    RECORD_SEC = 3  # 3 seconds recording
    MIN_SEC    = 0.5   # Minimum duration for playback
    CHUNK_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chunks')
    CHUNK_MS   = 1000  # 200ms chunks
    CHUNK_WAIT = 0.001  # 10ms wait between checks
    CHUNK_TIMEOUT = 1.0  # 1 second timeout for chunks
    WRITE_WAIT = 0.2   # Time to wait for file write

    # Create chunks directory if it doesn't exist
    if not os.path.exists(CHUNK_DIR):
        os.makedirs(CHUNK_DIR)

    print(f"[AUDIO] Using device: {CARD_DEV}")
    print(f"[AUDIO] Sample rate: {RATE}")
    print(f"[AUDIO] Period size: {PERIOD}")
    print(f"[AUDIO] Buffer size: {BUFFER}")
    print(f"[AUDIO] Recording will be saved to: {TMP_WAV}")

    # Initialize recording state
    recording_start = 0
    recording_chunks = []
    current_chunk = 0
    last_chunk_time = 0

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"[AUDIO] Status: {status}")
        recording_data.append(indata.copy())

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
            time.sleep(1.0/LED_COUNT)
        time.sleep(0.2)

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
        ev = dev.read_one()
        if ev and ev.type == ecodes.EV_KEY:
            key = categorize(ev)
            if key.scancode == ecodes.KEY_PLAYPAUSE:
                if key.keystate == key.key_down and rec_proc is None:
                    press_t = time.monotonic()
                    recording_start = press_t
                    print(f"[DEBUG] Button pressed at {press_t:.3f}")
                    # Start recording with fixed duration
                    cmd = ['arecord', '-D', CARD_DEV, '-r', RATE, '-c', '1', '-f', 'S16_LE',
                           '--period-size', PERIOD, '--buffer-size', BUFFER, '-d', str(RECORD_SEC), TMP_WAV]
                    rec_proc = subprocess.Popen(cmd)
                    state = 'recording'
                    print("[DEBUG] Started recording")
                elif key.keystate == key.key_up and state == 'recording':
                    duration = time.monotonic() - recording_start
                    print(f"[DEBUG] Button released after {duration:.3f} seconds")
                    # Let recording continue for full duration
                    if rec_proc:
                        print("[DEBUG] Waiting for recording to complete...")
                        try:
                            rec_proc.wait(timeout=RECORD_SEC + 1)  # Wait up to 4 seconds
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
                        state = 'processing'
                        processing_end = time.monotonic() + 0.5

        # Button-triggered states
        if GPIO.input(RED_BTN) == GPIO.LOW:
            state = 'manual_red'
        elif GPIO.input(GRN_BTN) == GPIO.LOW:
            state = 'manual_green'
        elif GPIO.input(BLU_BTN) == GPIO.LOW:
            state = 'manual_blue'

        now = time.monotonic()
        breath_phase += 0.05

        if state == 'idle':
            if LEDS_OK:
                b = 0.5 + 0.5 * math.sin(breath_phase)
                set_brightness(LED_BRIGHT * b)
                for i in range(strip.numPixels()):
                    if (i + idle_offset) % 4 == 0:
                        strip.setPixelColor(i, Color(*BLUE))
                    else:
                        strip.setPixelColor(i, Color(0,0,0))
                strip.show()
                idle_offset = (idle_offset + 1) % 4
            time.sleep(0.1)

        elif state == 'recording':
            if LEDS_OK:
                phase = math.sin(breath_phase * 2)
                for i in range(strip.numPixels()):
                    if phase > 0:
                        r, g, b = [int(c * (0.5 + 0.5 * phase)) for c in BLUE]
                    else:
                        r, g, b = [int(c * (0.5 - 0.5 * phase)) for c in GREEN]
                    strip.setPixelColor(i, Color(r, g, b))
                strip.show()
            time.sleep(0.05)

        elif state == 'processing':
            if LEDS_OK:
                prog = max(0, min(1, (processing_end - now) / 0.5))
                for i in range(strip.numPixels()):
                    r = int(BLUE[0] * prog + GREEN[0] * (1 - prog))
                    g = int(BLUE[1] * prog + GREEN[1] * (1 - prog))
                    b = int(BLUE[2] * prog + GREEN[2] * (1 - prog))
                    strip.setPixelColor(i, Color(r, g, b))
                strip.show()
            if now >= processing_end:
                print("[DEBUG] Starting playback...")
                play_cmd = [
                    'aplay', '-D', CARD_DEV,
                    '-r', RATE,
                    '-c', '1',
                    '-f', 'S16_LE',
                    '--period-size', PERIOD,
                    '--buffer-size', BUFFER,
                    '-q', TMP_WAV
                ]
                print(f"[DEBUG] Playback command: {' '.join(play_cmd)}")
                play_proc = subprocess.Popen(play_cmd)
                state = 'speaking'
                speak_phase = 0.0
            else:
                time.sleep(0.05)

        elif state == 'speaking':
            if LEDS_OK:
                b = 0.3 + 0.2 * math.sin(speak_phase * 5)
                set_brightness(LED_BRIGHT * b)
                for i in range(strip.numPixels()):
                    strip.setPixelColor(i, Color(*RED))
                strip.show()
                speak_phase += 0.1
            if play_proc and play_proc.poll() is not None:
                play_proc.wait()
                print("[DEBUG] Playback completed")
                set_brightness(LED_BRIGHT)
                state = 'idle'
            else:
                time.sleep(0.02)

        elif state == 'manual_red':
            show_color(RED)
            time.sleep(0.1)

        elif state == 'manual_green':
            show_color(GREEN)
            time.sleep(0.1)

        elif state == 'manual_blue':
            show_color(BLUE)
            time.sleep(0.1)

        else:
            state = 'idle'

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