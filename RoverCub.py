#!/usr/bin/env python3
"""
RoverCub Lite - Voice-Controlled AI Assistant

Features:
- Voice input with LLM responses
- Push-to-talk functionality
- Text-to-speech output
- Simple event handling
"""

import os
import time
import subprocess
import argparse
import openai
import requests
import tiktoken
from evdev import InputDevice, categorize, ecodes
import numpy as np
from rpi_ws281x import PixelStrip, Color
import threading
import queue
import json
import random

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/GPT-4 encoding

def tokenize_offline(text, max_tokens=32):
    return enc.encode(text)[:max_tokens]

# Global variables
client = None
CUSTOM_SYSTEM_MESSAGE = ""  # Empty by default, can be appended to assistant's message

# OpenAI Configuration
ASSISTANTS = {
    "1": {
        "name": "CodeMusAI",
        "id": "asst_2JwliRltBhXtZxrAU50U2bue",
        "description": "AI focused on music and coding",
        "thread_id": "thread_xiPkeFKCHDH5dZzq8aH4x5Ub",
        "voice": "alloy"  # Technical, precise voice
    },
    "2": {
        "name": "True Janet",
        "id": "asst_OzV6G6tLpeSO5yTIAXolrItL",
        "description": "AI with Janet's personality",
        "thread_id": "thread_j7irk1X6JFRbGnpSuvWT4gO1",
        "voice": "nova"  # Warm, friendly voice
    },
    "3": {
        "name": "Chris",
        "id": "asst_yQRINZO1rPiMnrMNn0XQJ7Ab",
        "description": "AI with Chris's personality",
        "thread_id": "thread_Z4z9zHEomQCMxAM8uTROB6Ew",
        "voice": "echo"  # Deep, authoritative voice
    },
    "4": {
        "name": "Eddie Mora",
        "id": "asst_P4YlCVBvgbi86A7Nx3YWl8gl",
        "description": "AI with Eddie's personality",
        "thread_id": "thread_FzGUQMAPjdRVWBYWOGXrO3qz",
        "voice": "fable"  # Storytelling voice
    },
    "5": {
        "name": "Justin Trudeau",
        "id": "asst_2IiLu41KEsr9BXCd8Xriq51o",
        "description": "AI with Justin Trudeau's diplomatic and progressive personality",
        "thread_id": None,
        "voice": "nova"  # Warm, diplomatic voice
    },
    "6": {
        "name": "Donald Trump",
        "id": "asst_Yg7zMD9fxNRJIMSdigSINkQN",
        "description": "AI with Donald Trump's bold and direct personality",
        "thread_id": None,
        "voice": "echo"  # Strong, assertive voice
    },
    "7": {
        "name": "Dale Carnegie",
        "id": "asst_zS5oxHoNfuL2avABzYv0Tqot",
        "description": "AI with Dale Carnegie's wisdom on human relations and leadership",
        "thread_id": None,
        "voice": "fable"  # Wise, storytelling voice
    },
    "8": {
        "name": "Joseph Smith",
        "id": "asst_Uh2N34I9BXlaSgyfEb0Qp1vs",
        "description": "AI with Joseph Smith's spiritual and historical perspective",
        "thread_id": None,
        "voice": "echo"  # Deep, spiritual voice
    },
    "9": {
        "name": "Pengu the Penguin",
        "id": "asst_yPqhexeP3RCwegMn5ZsR8AVj",
        "description": "AI with Pengu's playful and adventurous personality",
        "thread_id": None,
        "voice": "nova"  # Playful, friendly voice
    },
    "10": {
        "name": "Good Janet",
        "id": "asst_KW1nvrZGdF06eShdbtO5Wp2Z",
        "description": "AI with Good Janet's helpful and optimistic personality",
        "thread_id": None,
        "voice": "nova"  # Sweet, helpful voice
    },
    "11": {
        "name": "Bad Janet",
        "id": "asst_llaIAh7H2tMnekjn6y7i2Pm0",
        "description": "AI with Bad Janet's sassy and rebellious personality",
        "thread_id": None,
        "voice": "alloy"  # Sharp, sassy voice
    }
}
ASSISTANT_ID = None  # Will be set after user selection
ASSISTANT_MODEL = "gpt-4-turbo"  # Model to use for the assistant

# Conversation History
conversation_history = []
MAX_HISTORY_LENGTH = 10  # Maximum number of exchanges to remember

# OpenAI Thread Management
current_thread = None

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rovercub_config.json')

def load_config():
    """Load configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Update thread IDs in ASSISTANTS
                for key, assistant in ASSISTANTS.items():
                    if key in config.get('thread_ids', {}):
                        assistant['thread_id'] = config['thread_ids'][key]
                print("[CONFIG] Loaded thread IDs from config file")
                return config
    except Exception as e:
        print(f"[WARNING] Failed to load config: {e}")
    return {'thread_ids': {}}

def save_config():
    """Save configuration to file"""
    try:
        config = {
            'thread_ids': {
                key: assistant['thread_id'] 
                for key, assistant in ASSISTANTS.items() 
                if assistant.get('thread_id')
            }
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print("[CONFIG] Saved thread IDs to config file")
    except Exception as e:
        print(f"[WARNING] Failed to save config: {e}")

def add_to_history(role, content):
    """Add a message to the conversation history"""
    global conversation_history
    conversation_history.append({"role": role, "content": content})
    # Keep only the last N exchanges
    if len(conversation_history) > MAX_HISTORY_LENGTH * 2:  # *2 because each exchange has user and assistant
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH * 2:]

def clear_conversation_history():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = []
    print("[CONVERSATION] History cleared")

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
led_queue = queue.Queue()
led_thread = None
led_running = False

# Animation States
IDLE = 'idle'
LISTENING = 'listening'
PROCESSING = 'processing'
TALKING = 'talking'
HEARTBEAT = 'heartbeat'
SPECTRUM = 'spectrum'

# Animation Configuration
ANIMATION_CONFIG = {
    'processing': 'matrix',  # Can be 'matrix', 'pixel_rain', or 'rainbow'
    'snake_colors': True,    # Use token colors for snake
}

# ROYGBIV Color Palette with variations
TOKEN_COLORS = [
    (64, 0, 0),     # Red
    (64, 32, 0),    # Orange
    (64, 64, 0),    # Yellow
    (0, 64, 0),     # Green
    (0, 32, 64),    # Blue
    (32, 0, 64),    # Indigo
    (64, 0, 32),    # Violet
    (64, 0, 0),     # Red (darker)
    (48, 24, 0),    # Orange (darker)
    (48, 48, 0),    # Yellow (darker)
    (0, 48, 0),     # Green (darker)
    (0, 24, 48),    # Blue (darker)
    (24, 0, 48),    # Indigo (darker)
    (48, 0, 24),    # Violet (darker)
    (32, 0, 0),     # Red (darkest)
    (32, 16, 0),    # Orange (darkest)
    (32, 32, 0),    # Yellow (darkest)
    (0, 32, 0),     # Green (darkest)
    (0, 16, 32),    # Blue (darkest)
    (16, 0, 32),    # Indigo (darkest)
    (32, 0, 16),    # Violet (darkest)
]

def blend_colors(color1, color2, ratio):
    """Blend two colors with a given ratio"""
    return tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))

def get_token_color(token_value, phase):
    """Get a color for a token based on its value and animation phase"""
    # Use token value to select base color from ROYGBIV palette
    base_color = TOKEN_COLORS[token_value % len(TOKEN_COLORS)]
    
    # Add subtle animation
    pulse = 0.1 * np.sin(phase + token_value * 0.5)
    return tuple(int(c * (0.8 + pulse)) for c in base_color)

def visualize_tokens(tokens, phase):
    """Visualize tokens with gradual twinkling effect, starting from black"""
    if not LEDS_OK:
        return
    
    try:
        # Create a grid to store multiple tokens per LED
        token_grid = [[] for _ in range(LED_COUNT)]
        
        # Distribute tokens across the grid
        for i, token in enumerate(tokens):
            led_index = i % LED_COUNT
            token_grid[led_index].append(token)
        
        # Calculate how many LEDs should be lit based on phase
        # Gradually increase the number of lit LEDs
        progress = min(1.0, phase * 0.2)  # Slower progression
        num_lit = int(LED_COUNT * progress)
        
        # Update LEDs with gradual twinkling
        for i in range(LED_COUNT):
            if i >= num_lit:
                # Not yet time for this LED to be lit
                strip.setPixelColor(i, Color(0, 0, 0))
                continue
            
            if not token_grid[i]:
                # No tokens for this LED
                strip.setPixelColor(i, Color(0, 0, 0))
                continue
            
            # If multiple tokens, cycle through them
            if len(token_grid[i]) > 1:
                cycle_speed = 1.0  # Slower cycling
                token_index = int((phase * cycle_speed) % len(token_grid[i]))
                token = token_grid[i][token_index]
            else:
                token = token_grid[i][0]
            
            # Get base color from token value
            base_color = TOKEN_COLORS[token % len(TOKEN_COLORS)]
            
            # Random chance to twinkle (using phase + position for variation)
            twinkle_chance = np.sin(phase * 0.5 + i * 0.1) * 0.5 + 0.5  # Slower twinkling
            if np.random.random() < twinkle_chance:
                # This LED will twinkle
                twinkle_intensity = np.random.random()  # Random intensity for this twinkle
                red = int(64 * (0.5 + 0.5 * twinkle_intensity))  # Always some red component
                
                # Blend between base color and red based on twinkle intensity
                if twinkle_intensity > 0.5:
                    # More towards base color
                    color = blend_colors((red, 0, 0), base_color, (twinkle_intensity - 0.5) * 2)
                else:
                    # More towards red
                    color = blend_colors(base_color, (red, 0, 0), twinkle_intensity * 2)
            else:
                # No twinkle, use base color
                color = base_color
            
            strip.setPixelColor(i, Color(*color))
    except Exception as e:
        print(f"[LED] Error in token visualization: {e}")

def breathing_animation(phase):
    """Create a calm blue and green wave effect for idle state"""
    if not LEDS_OK:
        return
    
    try:
        # Calculate wave intensity
        intensity = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(phase))
        
        # Create a wave pattern with blue and green
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            wave = np.sin(phase + x * 0.5) * np.cos(phase + y * 0.5)
            
            # Alternate between blue and green waves
            if wave > 0.5:
                color = (0, int(64 * intensity), int(32 * intensity))  # More green
            elif wave > 0:
                color = (0, int(32 * intensity), int(64 * intensity))  # More blue
            else:
                color = (0, int(16 * intensity), int(16 * intensity))  # Dark blue-green
            
            strip.setPixelColor(i, Color(*color))
    except Exception as e:
        print(f"[LED] Error in breathing animation: {e}")

def scanning_animation(phase):
    """Create a dynamic RGB pattern for listening state"""
    if not LEDS_OK:
        return
    
    try:
        # Create a dynamic RGB pattern
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            
            # Calculate position-based color
            pos_phase = (x + y) * 0.5 + phase
            r = int(64 * (0.5 + 0.5 * np.sin(pos_phase)))
            g = int(64 * (0.5 + 0.5 * np.sin(pos_phase + 2.094)))  # 2π/3
            b = int(64 * (0.5 + 0.5 * np.sin(pos_phase + 4.189)))  # 4π/3
            
            # Add some movement
            movement = np.sin(phase * 2 + i * 0.1) * 0.2
            r = int(r * (1 + movement))
            g = int(g * (1 + movement))
            b = int(b * (1 + movement))
            
            strip.setPixelColor(i, Color(r, g, b))
    except Exception as e:
        print(f"[LED] Error in scanning animation: {e}")

def game_of_life_animation(grid, phase):
    """Create a dynamic Game of Life animation for processing state with blue and green pattern"""
    if not LEDS_OK:
        return grid
    
    try:
        # Initialize grid if None
        if grid is None:
            grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
            # Add some initial random cells
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if np.random.random() < 0.3:  # 30% chance of live cell
                        grid[y][x] = 1
        
        # Update grid based on Game of Life rules
        new_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Count live neighbors with wrap-around
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = (y + dy) % GRID_HEIGHT, (x + dx) % GRID_WIDTH
                        neighbors += grid[ny][nx]
                
                # Apply Game of Life rules
                if grid[y][x]:
                    new_grid[y][x] = 1 if neighbors in [2, 3] else 0
                else:
                    new_grid[y][x] = 1 if neighbors == 3 else 0
        
        # Update LEDs with blue and green colors
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                idx = y * GRID_WIDTH + x
                
                if new_grid[y][x]:
                    # Create dynamic blue and green color based on neighbors and phase
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = (y + dy) % GRID_HEIGHT, (x + dx) % GRID_WIDTH
                            neighbors += new_grid[ny][nx]
                    
                    # Use neighbors count to influence colors
                    base_intensity = 0.5 + 0.5 * np.sin(phase + neighbors * 0.5)
                    blue = int(64 * base_intensity)
                    green = int(64 * (1 - base_intensity))
                    strip.setPixelColor(idx, Color(0, green, blue))
                else:
                    # Add subtle glow for dead cells based on neighbors
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            ny, nx = (y + dy) % GRID_HEIGHT, (x + dx) % GRID_WIDTH
                            neighbors += new_grid[ny][nx]
                    
                    if neighbors > 0:
                        # Subtle blue-green glow based on number of neighbors
                        glow = min(16, neighbors * 4)  # Max brightness of 16
                        strip.setPixelColor(idx, Color(0, glow, glow))
                    else:
                        strip.setPixelColor(idx, Color(0, 0, 0))
        
        return new_grid
    except Exception as e:
        print(f"[LED] Error in Game of Life animation: {e}")
        return grid

def update_leds(state, phase, game_grid=None, tokens=None):
    """Update LEDs based on current state"""
    if not LEDS_OK:
        return game_grid
    
    try:
        if state == IDLE:
            breathing_animation(phase)  # Calm breathing animation
        elif state == LISTENING:
            spectrum_animation(phase)  # Spectrum loading animation
        elif state == HEARTBEAT:
            heartbeat_animation(phase)  # Heartbeat animation
        elif state == PROCESSING:
            if ANIMATION_CONFIG['processing'] == 'matrix':
                matrix_animation(phase)
            elif ANIMATION_CONFIG['processing'] == 'pixel_rain':
                pixel_rain_animation(phase)
            else:  # rainbow
                rainbow_animation(phase)
        elif state == TALKING and tokens is not None:
            if ANIMATION_CONFIG['snake_colors']:
                snake_animation_with_tokens(phase, tokens)  # Snake with token colors
            else:
                visualize_tokens(tokens, phase)  # Original token visualization
        else:
            # Fallback to breathing animation if state is unknown
            breathing_animation(phase)
        
        # Always show the LEDs
        strip.show()
        return game_grid
    except Exception as e:
        print(f"[LED] Error updating LEDs: {e}")
        # Fallback to breathing animation on error
        breathing_animation(phase)
        strip.show()
        return game_grid

def spectrum_animation(phase):
    """Create a spectrum loading animation"""
    if not LEDS_OK:
        return
    
    try:
        # Create a dynamic spectrum pattern
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            
            # Calculate position-based color
            pos_phase = (x + y) * 0.5 + phase
            r = int(64 * (0.5 + 0.5 * np.sin(pos_phase)))
            g = int(64 * (0.5 + 0.5 * np.sin(pos_phase + 2.094)))  # 2π/3
            b = int(64 * (0.5 + 0.5 * np.sin(pos_phase + 4.189)))  # 4π/3
            
            # Add some movement
            movement = np.sin(phase * 2 + i * 0.1) * 0.2
            r = int(r * (1 + movement))
            g = int(g * (1 + movement))
            b = int(b * (1 + movement))
            
            strip.setPixelColor(i, Color(r, g, b))
    except Exception as e:
        print(f"[LED] Error in spectrum animation: {e}")

def heartbeat_animation(phase):
    """Create a heartbeat animation"""
    if not LEDS_OK:
        return
    
    try:
        # Calculate heartbeat pattern
        beat = np.sin(phase * 4) * 0.5 + 0.5  # Faster pulse
        intensity = 0.3 + 0.7 * beat
        
        # Create a pulsing red pattern
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            
            # Add some variation based on position
            pos_factor = np.sin(x * 0.5 + y * 0.5) * 0.2 + 0.8
            r = int(64 * intensity * pos_factor)
            g = int(16 * intensity * pos_factor)
            b = int(16 * intensity * pos_factor)
            
            strip.setPixelColor(i, Color(r, g, b))
    except Exception as e:
        print(f"[LED] Error in heartbeat animation: {e}")

def matrix_animation(phase):
    """Create a Matrix-style animation"""
    if not LEDS_OK:
        return
    
    try:
        # Create falling green characters effect
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            
            # Calculate falling effect
            fall_phase = (x * 0.5 + phase) % (2 * np.pi)
            brightness = np.sin(fall_phase) * 0.5 + 0.5
            
            # Add some random variation
            if np.random.random() < 0.1:  # 10% chance for brighter pixel
                brightness = 1.0
            
            # Set green color with varying brightness
            g = int(64 * brightness)
            strip.setPixelColor(i, Color(0, g, 0))
    except Exception as e:
        print(f"[LED] Error in matrix animation: {e}")

def pixel_rain_animation(phase):
    """Create a pixel rain animation"""
    if not LEDS_OK:
        return
    
    try:
        # Create falling pixels effect
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            
            # Calculate falling effect
            fall_phase = (x * 0.5 + phase) % (2 * np.pi)
            brightness = np.sin(fall_phase) * 0.5 + 0.5
            
            # Random color for each pixel
            r = int(64 * brightness * np.random.random())
            g = int(64 * brightness * np.random.random())
            b = int(64 * brightness * np.random.random())
            
            strip.setPixelColor(i, Color(r, g, b))
    except Exception as e:
        print(f"[LED] Error in pixel rain animation: {e}")

def rainbow_animation(phase):
    """Create a rainbow animation"""
    if not LEDS_OK:
        return
    
    try:
        # Create rainbow pattern
        for i in range(LED_COUNT):
            x = i % GRID_WIDTH
            y = i // GRID_WIDTH
            
            # Calculate color based on position and phase
            hue = (x + y + phase) * 0.1
            r = int(64 * (0.5 + 0.5 * np.sin(hue)))
            g = int(64 * (0.5 + 0.5 * np.sin(hue + 2.094)))  # 2π/3
            b = int(64 * (0.5 + 0.5 * np.sin(hue + 4.189)))  # 4π/3
            
            strip.setPixelColor(i, Color(r, g, b))
    except Exception as e:
        print(f"[LED] Error in rainbow animation: {e}")

def snake_animation_with_tokens(phase, tokens):
    """Create a snake animation using token colors"""
    if not LEDS_OK:
        return
    
    try:
        # Initialize snake if needed
        if not hasattr(snake_animation_with_tokens, 'snake'):
            snake_animation_with_tokens.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
            snake_animation_with_tokens.direction = (1, 0)
            snake_animation_with_tokens.last_move = 0
            snake_animation_with_tokens.move_interval = 0.2
        
        # Move snake based on time
        current_time = time.time()
        if current_time - snake_animation_with_tokens.last_move >= snake_animation_with_tokens.move_interval:
            # Calculate new head position
            head_x, head_y = snake_animation_with_tokens.snake[0]
            dx, dy = snake_animation_with_tokens.direction
            new_x = (head_x + dx) % GRID_WIDTH
            new_y = (head_y + dy) % GRID_HEIGHT
            
            # Check for collisions with self
            if (new_x, new_y) in snake_animation_with_tokens.snake:
                # Reset snake
                snake_animation_with_tokens.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
                snake_animation_with_tokens.direction = (1, 0)
            else:
                # Move snake
                snake_animation_with_tokens.snake.insert(0, (new_x, new_y))
                if len(snake_animation_with_tokens.snake) > len(tokens):
                    snake_animation_with_tokens.snake.pop()
            
            snake_animation_with_tokens.last_move = current_time
        
        # Clear all LEDs
        for i in range(LED_COUNT):
            strip.setPixelColor(i, Color(0, 0, 0))
        
        # Draw snake with token colors
        for i, (x, y) in enumerate(snake_animation_with_tokens.snake):
            if i < len(tokens):
                # Get color from token
                token = tokens[i]
                base_color = TOKEN_COLORS[token % len(TOKEN_COLORS)]
            else:
                # Default to white if no token color
                base_color = (64, 64, 64)
            
            # Calculate LED index
            led_index = y * GRID_WIDTH + x
            strip.setPixelColor(led_index, Color(*base_color))
    except Exception as e:
        print(f"[LED] Error in snake animation: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='RoverCub Lite - Voice-Controlled AI Assistant')
    parser.add_argument('--test', action='store_true', help='Run audio test')
    parser.add_argument('--text', action='store_true', help='Use text input instead of voice')
    parser.add_argument('--device', default='plughw:1,0', help='ALSA device')
    parser.add_argument('--rate', type=int, default=48000, help='Sample rate')
    parser.add_argument('--period', type=int, default=6000, help='Period size')
    return parser.parse_args()

def setup_openai():
    global client
    API_KEY = ""
    client = openai.OpenAI(api_key=API_KEY)
    print("[INFO] Using OpenAI API")

def select_assistant():
    """Let user select an assistant"""
    global ASSISTANT_ID, current_thread
    
    try:
        # First verify we can access the assistants
        assistants = client.beta.assistants.list()
        available_assistants = {a.id: a for a in assistants.data}
        
        print("\n[CONFIG] Available Assistants:")
        found_assistants = False
        for key, assistant in ASSISTANTS.items():
            if assistant["id"] in available_assistants:
                thread_status = "Active" if assistant["thread_id"] else "New"
                print(f"{key}. {assistant['name']} - {assistant['description']} ({thread_status})")
                found_assistants = True
            else:
                print(f"{key}. {assistant['name']} - NOT FOUND (ID: {assistant['id']})")
        
        if not found_assistants:
            print("\n[ERROR] No assistants found! Please check:")
            print("1. Your API key is correct")
            print("2. The assistant IDs are correct")
            print("3. You have access to these assistants")
            print("\nWould you like to:")
            print("1. List all available assistants")
            print("2. Exit")
            choice = input("Enter your choice (1-2): ").strip()
            if choice == "1":
                print("\n[INFO] Your available assistants:")
                for a in assistants.data:
                    print(f"- {a.name} (ID: {a.id})")
            return False
        
        while True:
            choice = input("\nSelect an assistant (1-11): ").strip()
            if choice in ASSISTANTS:
                if ASSISTANTS[choice]["id"] in available_assistants:
                    ASSISTANT_ID = ASSISTANTS[choice]["id"]
                    # Set current thread to assistant's thread if it exists
                    current_thread = ASSISTANTS[choice]["thread_id"]
                    print(f"\n[ASSISTANT] Selected {ASSISTANTS[choice]['name']}")
                    if current_thread:
                        print(f"[THREAD] Using existing thread: {current_thread}")
                    return True
                else:
                    print(f"[ERROR] Assistant {ASSISTANTS[choice]['name']} not found!")
            print("Invalid choice. Please select 1-11.")
    except Exception as e:
        print(f"[ERROR] Failed to access assistants: {e}")
        print("\n[INFO] Please check your API key and permissions")
        return False

def create_new_thread():
    """Create a new OpenAI thread"""
    global current_thread, ASSISTANT_ID
    
    try:
        # Find the assistant that matches the current ASSISTANT_ID
        for key, assistant in ASSISTANTS.items():
            if assistant["id"] == ASSISTANT_ID:
                # Create new thread
                thread = client.beta.threads.create()
                # Store thread ID in assistant's data
                assistant["thread_id"] = thread.id
                current_thread = thread.id
                print(f"[THREAD] Created new conversation thread with ID: {thread.id}")
                print(f"[THREAD] Thread ID stored for {assistant['name']}")
                # Save the new thread ID to config
                save_config()
                return thread
    except Exception as e:
        print(f"[ERROR] Failed to create thread: {e}")
        return None

def get_current_assistant_key():
    """Get the key of the currently selected assistant"""
    for key, assistant in ASSISTANTS.items():
        if assistant["id"] == ASSISTANT_ID:
            return key
    return None

def switch_assistant():
    """Randomly switch to a different assistant"""
    global ASSISTANT_ID, current_thread
    
    try:
        current_key = get_current_assistant_key()
        available_keys = [k for k in ASSISTANTS.keys() if k != current_key]
        
        if not available_keys:
            print("[WARNING] No other assistants available")
            return False
        
        # Randomly select a new assistant
        new_key = random.choice(available_keys)
        new_assistant = ASSISTANTS[new_key]
        
        # Update global variables
        ASSISTANT_ID = new_assistant["id"]
        current_thread = new_assistant["thread_id"]
        
        # Announce the change
        announcement = f"Switching to {new_assistant['name']}. {new_assistant['description']}"
        print(f"[ASSISTANT] {announcement}")
        
        # Use TTS to announce the change
        tts_file = text_to_speech(announcement, voice=new_assistant["voice"])
        if tts_file:
            play_audio(tts_file)
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to switch assistant: {e}")
        return False

def get_ai_response(text, use_openai=True):
    """Get response from AI using OpenAI thread"""
    global current_thread
    
    # Check for assistant switch command
    if text.lower().strip().startswith("change assistant"):
        if switch_assistant():
            return "Assistant switched successfully.", []
        else:
            return "Failed to switch assistant.", []
    
    if use_openai:
        try:
            # Create thread if none exists
            if current_thread is None:
                thread = create_new_thread()
                if thread is None:
                    raise Exception("Failed to create thread")
            else:
                print(f"[THREAD] Using existing thread: {current_thread}")
            
            # Add message to thread
            message = client.beta.threads.messages.create(
                thread_id=current_thread,
                role="user",
                content=text
            )
            print(f"[THREAD] Added message to thread: {current_thread}")
            
            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=current_thread,
                assistant_id=ASSISTANT_ID
            )
            print(f"[THREAD] Started run: {run.id}")
            
            # Wait for completion
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=current_thread,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    break
                elif run_status.status == 'failed':
                    raise Exception("Run failed")
                time.sleep(0.5)
            
            # Get the latest message
            messages = client.beta.threads.messages.list(
                thread_id=current_thread,
                order='desc',
                limit=1
            )
            response_text = messages.data[0].content[0].text.value
            
            # Get tokens for visualization
            tokens = tokenize_offline(response_text)
            print(f"[TOKENS] Generated {len(tokens)} tokens")
            return response_text, tokens
            
        except Exception as e:
            print(f"[ERROR] OpenAI thread error: {e}")
            print("[INFO] Falling back to chat completion API")
            # Fallback to regular chat completion with the correct assistant
            try:
                # Get the assistant's instructions
                assistant = client.beta.assistants.retrieve(ASSISTANT_ID)
                system_message = assistant.instructions if assistant.instructions else ""
                
                response = client.chat.completions.create(
                    model=ASSISTANT_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": text}
                    ]
                )
                response_text = response.choices[0].message.content
                tokens = tokenize_offline(response_text)
                print(f"[TOKENS] Generated {len(tokens)} tokens")
                return response_text, tokens
            except Exception as e2:
                print(f"[ERROR] Chat completion fallback failed: {e2}")
                raise
    else:
        # LM Studio fallback
        response = requests.post(
            "http://10.0.0.105:1234/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": text}
                ],
                "temperature": 0.9,
                "max_tokens": 150
            }
        )
        response_text = response.json()["choices"][0]["message"]["content"]
        tokens = tokenize_offline(response_text)
        print(f"[TOKENS] Generated {len(tokens)} tokens")
        return response_text, tokens

def safe_print(text):
    """Safely print text that may contain Unicode characters"""
    try:
        print(text)
    except UnicodeEncodeError:
        # If we can't print the Unicode text, print a sanitized version
        sanitized = text.encode('ascii', 'replace').decode('ascii')
        print(sanitized)

def text_to_speech(text, voice=None):
    """Convert text to speech using the specified voice"""
    print("[TTS] Converting response to speech...")
    try:
        # Get the current assistant's voice if none specified
        if voice is None:
            current_key = get_current_assistant_key()
            if current_key:
                voice = ASSISTANTS[current_key]["voice"]
            else:
                voice = "shimmer"  # Default voice
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        tts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tts_response.mp3')
        
        # Use the recommended streaming response method
        with open(tts_file, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        return tts_file
    except Exception as e:
        print(f"[ERROR] TTS conversion failed: {e}")
        return None

def set_audio_volume():
    """Set the audio volume once at startup"""
    print("[VOLUME] Setting initial volume...")
    try:
        # Try different common volume control names
        volume_controls = ['Master', 'PCM', 'Speaker', 'Headphone']
        volume_set = False
        
        for control in volume_controls:
            try:
                # Try to set volume to 70%
                result = subprocess.run(['amixer', '-D', 'hw:1', 'sset', control, '91%'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"[VOLUME] Set {control} volume to 91%")
                    volume_set = True
                    break
            except Exception as e:
                continue
        
        if not volume_set:
            print("[WARNING] Could not set volume, continuing with default volume")
    except Exception as e:
        print(f"[ERROR] Volume setting failed: {e}")

def play_audio(file_path):
    print("[PLAY] Playing audio response...")
    try:
            # Play the audio file
        play_cmd = ['mpg123', '-q', file_path]
        subprocess.run(play_cmd)
            
        # Clean up the file
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[WARNING] Could not remove temporary file: {e}")

    except Exception as e:
        print(f"[ERROR] Audio playback failed: {e}")

def verify_recording(file_path, min_size_kb=5):
    """Verify that the recording file exists and has a reasonable size"""
    if not os.path.exists(file_path):
        print("[ERROR] Recording file not found")
        return False
    
    size_kb = os.path.getsize(file_path) / 1024
    print(f"[INFO] Recording file size: {size_kb:.1f}KB")
    
    if size_kb < min_size_kb:
        print(f"[ERROR] Recording file too small ({size_kb:.1f}KB < {min_size_kb}KB)")
        return False
    
    # Add check for file permissions
    if not os.access(file_path, os.R_OK):
        print("[ERROR] Cannot read recording file - permission denied")
        return False
        
    # Add check for file content
    try:
        with open(file_path, 'rb') as f:
            header = f.read(44)  # WAV header is 44 bytes
            if not header.startswith(b'RIFF'):
                print("[ERROR] Invalid WAV file format")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to verify WAV file: {e}")
        return False
    
    return True

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

def check_audio_permissions():
    """Check if we have proper permissions for audio devices"""
    try:
        # Check if we can access the audio device
        test_cmd = ['arecord', '-l']
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[WARNING] Audio device access check failed. You may need to run with sudo.")
            print("[INFO] Try running: sudo python RoverCub_lite.py")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Permission check failed: {e}")
        return False

def safe_terminate_process(proc):
    """Safely terminate a process with proper cleanup"""
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except:
                pass
        except:
            pass

def initialize_leds():
    """Initialize the LED strip"""
    global LEDS_OK, strip
    try:
        strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                          LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        strip.begin()
        LEDS_OK = True
        print("[LED] Initialized successfully")
    except Exception as e:
        print(f"[LED] Initialization failed: {e}")
        LEDS_OK = False

def led_update_thread():
    """Thread function for continuous LED updates"""
    global led_running, strip, LEDS_OK
    
    phase = 0.0
    game_grid = None
    current_state = IDLE
    current_tokens = None
    
    while led_running:
        try:
            # Check for new state/tokens from queue
            try:
                while not led_queue.empty():
                    new_state, new_tokens = led_queue.get_nowait()
                    current_state = new_state
                    current_tokens = new_tokens
            except queue.Empty:
                pass
            
            # Update LEDs based on current state
            if current_state == TALKING and current_tokens is not None:
                update_leds(current_state, phase, game_grid, current_tokens)
            else:
                game_grid = update_leds(current_state, phase, game_grid, current_tokens)
            
            # Update animation phase
            phase += 0.1  # Adjust speed as needed
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
        except Exception as e:
            print(f"[LED] Error in LED thread: {e}")
            time.sleep(0.1)  # Longer sleep on error

def start_led_thread():
    """Start the LED update thread"""
    global led_thread, led_running, LEDS_OK
    
    if not LEDS_OK:
        print("[LED] LED initialization failed, cannot start thread")
        return
    
    if led_thread is None or not led_thread.is_alive():
        led_running = True
        led_thread = threading.Thread(target=led_update_thread, daemon=True)
        led_thread.start()
        print("[LED] LED update thread started")

def stop_led_thread():
    """Stop the LED update thread"""
    global led_thread, led_running
    
    if led_thread is not None and led_thread.is_alive():
        led_running = False
        led_thread.join(timeout=1.0)
        print("[LED] LED update thread stopped")

def update_led_state(state, tokens=None):
    """Update LED state and tokens through the queue"""
    try:
        led_queue.put((state, tokens))
    except Exception as e:
        print(f"[LED] Error updating LED state: {e}")

def clear_conversation():
    """Clear the current conversation thread"""
    global current_thread
    current_thread = None
    # Update config to remove the thread ID
    for key, assistant in ASSISTANTS.items():
        if assistant["id"] == ASSISTANT_ID:
            assistant["thread_id"] = None
            break
    save_config()
    print("[THREAD] Conversation thread cleared")

def handle_ai_response(ai_response, tokens):
    """Handle AI response callback"""
    try:
        # Immediately update LED state for talking with tokens
        update_led_state(TALKING, tokens)
        print(f"[STATE] Changed to {TALKING}")
        print(f"[TOKENS] Generated {len(tokens)} tokens")
        safe_print(f"[AI] Response: {ai_response}")
        
        # Convert to speech and play
        tts_file = text_to_speech(ai_response)
        play_audio(tts_file)
        
        # Return to idle state
        update_led_state(IDLE)
        print(f"[STATE] Changed to {IDLE}")
        return IDLE  # Explicitly return the new state
    except Exception as e:
        print(f"[ERROR] Error handling AI response: {e}")
        update_led_state(IDLE)
        print(f"[STATE] Changed to {IDLE}")
        return IDLE  # Return idle state even on error

def cleanup_leds():
    """Turn off all LEDs and clean up"""
    global LEDS_OK, strip
    if LEDS_OK and strip:
        try:
            # Turn off all LEDs
            for i in range(LED_COUNT):
                strip.setPixelColor(i, Color(0, 0, 0))
            strip.show()
            print("[LED] All LEDs turned off")
        except Exception as e:
            print(f"[ERROR] Failed to turn off LEDs: {e}")

def find_headset_path():
    """Find the correct event device path for the headset"""
    try:
        for event_file in os.listdir('/dev/input/by-id'):
            if 'usb-Walmart_AB13X_Headset_Adapter' in event_file:
                return os.path.join('/dev/input/by-id', event_file)
        return '/dev/input/event0'  # Default fallback
    except Exception as e:
        print(f"[ERROR] Error finding headset path: {e}")
        return '/dev/input/event0'

def initialize_headset():
    """Initialize the headset device"""
    try:
        EVENT_PATH = find_headset_path()
        dev = InputDevice(EVENT_PATH)
        print(f"[MIC] Found headset at {dev.path}")
        return dev
    except Exception as e:
        print(f"[ERROR] Could not initialize headset: {e}")
        print("Falling back to GPIO buttons only")
        return None

def main():
    global LEDS_OK, strip
    args = parse_args()
    
    # Load saved thread IDs
    load_config()
    
    # Initialize LEDs
    initialize_leds()
    if LEDS_OK:
        start_led_thread()
    
    # Check permissions first
    if not check_audio_permissions():
        return

    # Set volume once at startup
    set_audio_volume()

    if args.test:
        test_audio(args.device, args.rate, args.period)
        return

    # Start in spectrum loading state
    update_led_state(SPECTRUM)
    print(f"[STATE] Changed to {SPECTRUM}")

    # Prompt user for API choice
    print("\n[CONFIG] Please choose your AI provider:")
    print("1. OpenAI (requires API key)")
    print("2. LM Studio (local)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    while choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")
        choice = input("Enter your choice (1 or 2): ").strip()

    setup_openai()
    
    if choice == '1':
        if not select_assistant():
            print("[ERROR] Failed to select assistant. Exiting...")
            return
    else:
        print("[INFO] Using LM Studio at http://10.0.0.105:1234")

    # Switch to idle state after initialization
    update_led_state(IDLE)
    print(f"[STATE] Changed to {IDLE}")

    if args.text:
        print("\n[INFO] Running in text mode. Type your message and press Enter.")
        print("Type 'quit' to exit.")
        print("Type 'clear' to clear conversation thread.")
        while True:
            try:
                user_input = input("\nYour message: ").strip()
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    clear_conversation()
                    continue
                    
                print("\n[AI] Getting response...")
                update_led_state(PROCESSING)
                print(f"[STATE] Changed to {PROCESSING}")
                
                # Get AI response directly
                try:
                    ai_response, tokens = get_ai_response(user_input, choice == '1')
                    state = handle_ai_response(ai_response, tokens)  # Get new state from handler
                    # State will be set to IDLE by handle_ai_response
                except Exception as e:
                    print(f"[ERROR] Failed to get AI response: {e}")
                    state = IDLE
                    update_led_state(state)
                    print(f"[STATE] Changed to {IDLE}")
                
            except Exception as e:
                print(f"[ERROR] Text mode error: {e}")
                state = IDLE
                update_led_state(state)
                print(f"[STATE] Changed to {IDLE}")
                continue

    else:
        # Voice mode
        EVENT_PATH = find_headset_path()  # Get initial path
        CARD_DEV = args.device
        RATE = str(args.rate)
        PERIOD = str(args.period)
        BUFFER = str(int(PERIOD)*4)
        TMP_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptt_clip.wav')
        RECORD_SEC = 5
        RELEASE_DELAY = 0.5

        # Ensure the temporary directory exists and is writable
        try:
            os.makedirs(os.path.dirname(TMP_WAV), exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Could not create temp directory: {e}")
            TMP_WAV = '/tmp/ptt_clip.wav'
            print(f"[INFO] Using fallback path: {TMP_WAV}")

        state = IDLE  # Start in idle state
        rec_proc = None
        play_proc = None
        recording_start = 0.0
        button_pressed = False

        # Initialize headset once at startup
        dev = initialize_headset()
        if not dev:
            print("[WARNING] No headset found, continuing with GPIO buttons only")

        try:
            while True:
                # Handle button events
                if dev:
                    try:
                        ev = dev.read_one()
                        if ev and ev.type == ecodes.EV_KEY:
                            key = categorize(ev)
                            if key.scancode == ecodes.KEY_PLAYPAUSE:
                                if key.keystate == key.key_down and not button_pressed and state == IDLE:
                                    button_pressed = True
                                    recording_start = time.monotonic()
                                    print("[BUTTON] Button pressed")
                                    
                                    # Switch to heartbeat state
                                    state = HEARTBEAT
                                    update_led_state(state)
                                    print(f"[STATE] Changed to {state}")
                                    
                                    # Start recording
                                    print("[REC] Starting recording...")
                                    try:
                                        # Ensure any existing recording is stopped
                                        if rec_proc:
                                            safe_terminate_process(rec_proc)
                                            rec_proc = None
                                        
                                        # Use optimized settings for throat microphone
                                        rec_cmd = [
                                            'arecord',
                                            '-D', CARD_DEV,
                                            '-r', '16000',
                                            '-f', 'S16_LE',
                                            '-c', '1',
                                            '-t', 'wav',
                                            '-d', '10',
                                            '-v',
                                            '--buffer-size=32000',
                                            '--period-size=8000',
                                            TMP_WAV
                                        ]
                                        print(f"[REC] Running command: {' '.join(rec_cmd)}")
                                        rec_proc = subprocess.Popen(['sudo'] + rec_cmd)
                                        
                                        # Wait for recording to complete
                                        print("[REC] Recording for 5 seconds...")
                                        time.sleep(5)
                                        
                                        # Ensure recording is stopped
                                        if rec_proc:
                                            safe_terminate_process(rec_proc)
                                            rec_proc = None
                                        
                                        print("[REC] Recording completed")
                                        
                                        # Process the recording
                                        if os.path.exists(TMP_WAV):
                                            # Verify recording quality
                                            if not verify_recording(TMP_WAV):
                                                print("[ERROR] Recording quality check failed")
                                                state = IDLE
                                                update_led_state(state)
                                                print(f"[STATE] Changed to {state}")
                                                continue
                                            
                                            print("[STT] Converting speech to text...")
                                            try:
                                                with open(TMP_WAV, "rb") as audio_file:
                                                    transcript = client.audio.transcriptions.create(
                                                        model="whisper-1",
                                                        file=audio_file,
                                                        language="en",
                                                        response_format="text",
                                                        temperature=0.0,
                                                        prompt="This is a recording from a throat microphone. The audio is tinny and quieter than normal speech. Focus on the fundamental frequencies and ignore background noise."
                                                    )
                                                
                                                if isinstance(transcript, str):
                                                    transcript_text = transcript
                                                else:
                                                    transcript_text = transcript.text
                                                
                                                safe_print(f"[STT] User said: {transcript_text}")
                                                
                                                if len(transcript_text.strip()) < 2:
                                                    print("[WARNING] Transcription too short, might be incomplete")
                                                    state = IDLE
                                                    update_led_state(state)
                                                    print(f"[STATE] Changed to {state}")
                                                    continue

                                                print("[AI] Getting response...")
                                                state = PROCESSING
                                                update_led_state(state)
                                                print(f"[STATE] Changed to {PROCESSING}")
                                                
                                                # Get AI response directly
                                                try:
                                                    ai_response, tokens = get_ai_response(transcript_text, choice == '1')
                                                    state = handle_ai_response(ai_response, tokens)  # Get new state from handler
                                                    # State will be set to IDLE by handle_ai_response
                                                except Exception as e:
                                                    print(f"[ERROR] Failed to get AI response: {e}")
                                                    state = IDLE
                                                    update_led_state(state)
                                                    print(f"[STATE] Changed to {IDLE}")
                                                
                                            except Exception as e:
                                                print(f"[ERROR] Failed to process audio: {e}")
                                                state = IDLE
                                                update_led_state(state)
                                                print(f"[STATE] Changed to {IDLE}")
                                            
                                    except Exception as e:
                                        print(f"[ERROR] Failed to start recording: {e}")
                                        state = IDLE
                                        update_led_state(state)
                                        print(f"[STATE] Changed to {IDLE}")
                                    
                                elif key.keystate == key.key_up and button_pressed:
                                    button_pressed = False
                                    print("[BUTTON] Button released")
                                    if state == HEARTBEAT:
                                        state = IDLE
                                        update_led_state(state)
                                        print(f"[STATE] Changed to {state}")
                    except Exception as e:
                        print(f"[ERROR] Reading from device: {e}")
                        # Try to reinitialize the device
                        try:
                            dev.close()  # Close the old device first
                        except:
                            pass
                        dev = initialize_headset()
                        if not dev:
                            print("[WARNING] Could not reinitialize headset, continuing with GPIO buttons only")
                        time.sleep(0.1)

                # Small sleep to prevent CPU hogging
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[EXIT] Cleaning up...")
            # Flash screen before turning off
            try:
                for i in range(LED_COUNT):
                    strip.setPixelColor(i, Color(64, 0, 0))  # Red flash
                strip.show()
                time.sleep(0.1)
            except:
                pass
            
            if dev:
                try:
                    dev.close()
                except:
                    pass
            safe_terminate_process(rec_proc)
            safe_terminate_process(play_proc)
            if os.path.exists(TMP_WAV):
                try:
                    os.remove(TMP_WAV)
                except:
                    pass
            stop_led_thread()
            cleanup_leds()
            print("[DONE] Goodbye!")

if __name__ == "__main__":
    main()
