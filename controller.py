import pygame
import time
import serial

def deadzone(v,threshold=0.05):
    return 0 if abs(v) < threshold else v


def main():
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()

    ser = serial.Serial


    if count == 0:
        print("Controller not found")
        exit()

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print("Name:",joy.get_name())
    print("Buttons:",joy.get_numbuttons())
    print("Axes:",joy.get_numaxes())

    while True:
        for event in pygame.event.get():
            print(event)

        leftX = deadzone(joy.get_axis(0))
        leftY = deadzone(joy.get_axis(1))
        rightX = deadzone(joy.get_axis(3))
        rightY = deadzone(joy.get_axis(4))
        rotL = deadzone(joy.get_axis(2))
        rotR = deadzone(joy.get_axis(5))

        print(leftX)
        print(leftY)
        print(rotL)

        time.sleep(0.01)


#左     横：axes 0、縦：axes 1、ボタン：axes 2、L：button 4
#右     横：axes 3、縦：axes 4、ボタン：axes 5、R：button 5
#右十字 下から反時計回りに　button 0~3
#左十字 返り値が(,)で、正の方向が１、負の方向が‐1
#share : button 8
# options : button 9

if __name__ == "__main__":
    main()