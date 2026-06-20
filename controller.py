import pygame
import time
import serial

pygame.init()

screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("Controller Test")

pygame.joystick.init()

print("Joystick count:", pygame.joystick.get_count())

joy = pygame.joystick.Joystick(0)
joy.init()

print("Name:", joy.get_name())

while True:
    for event in pygame.event.get():
        pass

    print(
        "axis0 =", joy.get_axis(0),
        "axis1 =", joy.get_axis(1),
        "button0 =", joy.get_button(0)
    )

    time.sleep(0.1)