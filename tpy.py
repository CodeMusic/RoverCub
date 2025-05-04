import RPi.GPIO as GPIO
import time

BUTTON = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Add pull-up!

try:
    while True:
        state = GPIO.input(BUTTON)
        if state:
            print("Button NOT pressed (off)")
        else:
            print("Button PRESSED (on)")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("Exiting...")
    GPIO.cleanup()
