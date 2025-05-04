from gpiozero import Button, LED
from signal import pause

button = Button(17)
led_red = LED(27)
led_green = LED(22)

def on_press():
    led_red.toggle()
    led_green.toggle()

button.when_pressed = on_press
print("Waiting for button press...")
pause()
